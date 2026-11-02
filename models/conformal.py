"""Interval calibration: make the stated 80% band actually cover 80%.

The AFT model's `aft_loss_distribution_scale` is a training hyperparameter, not
a learned quantity, so nothing in fitting guarantees the resulting intervals
are honest. Measured out of the box, the nominal P10-P90 band covered 0.69 of
held-out outcomes. A confidence badge built on that would be lying, and the
whole uncertainty-to-safety-stock chain downstream would be mis-scaled.

Method: a single scalar width multiplier `s` applied to sigma, chosen on a
held-out calibration split so empirical coverage matches nominal. This is the
split-conformal idea reduced to its simplest useful form - MAPIE has no
survival mode, and a one-parameter correction is both defensible and
impossible to overfit with thousands of calibration rows.

Censoring is handled rather than ignored. Calibrating only on uncensored rows
would bias toward short-lived components (they are the ones that get observed).
A censored row still carries one-sided information: if the component was last
seen alive *past* its own P90, the interval definitely missed high. Those rows
are counted as known misses; censored rows that are still inside their band are
genuinely indeterminate and excluded.

Run:
    python -m models.conformal
    python -m models.conformal --report      # coverage curve only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

from models.survival_xgb import sigma_for, to_dmatrix

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

TARGET_COVERAGE = 0.80  # the P10-P90 band
NOMINAL_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
SEARCH = np.arange(0.60, 3.01, 0.02)


# --------------------------------------------------------------------------
# Coverage with censoring
# --------------------------------------------------------------------------


def coverage(
    mu: np.ndarray,
    sigma: np.ndarray,
    y_lower: np.ndarray,
    event: np.ndarray,
    level: float,
    multiplier: float = 1.0,
) -> tuple[float, int]:
    """Empirical coverage of a central `level` band, and the usable row count.

    Returns (coverage, n_used). Uncensored rows are scored normally. Censored
    rows count only when they are already known misses (last seen alive beyond
    the upper bound); otherwise they are indeterminate and dropped.
    """
    s = sigma * multiplier
    lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    lo = np.exp(mu + s * norm.ppf(lo_q))
    hi = np.exp(mu + s * norm.ppf(hi_q))

    obs = event.astype(bool)
    hit_obs = (y_lower >= lo) & (y_lower <= hi)

    # Censored and already past the upper bound -> definite miss.
    cens_miss = (~obs) & (y_lower > hi)
    n_used = int(obs.sum() + cens_miss.sum())
    if n_used == 0:
        return float("nan"), 0
    n_hit = int(hit_obs[obs].sum())  # censored misses contribute 0 hits
    return n_hit / n_used, n_used


def fit_multiplier(
    mu: np.ndarray,
    sigma: np.ndarray,
    y_lower: np.ndarray,
    event: np.ndarray,
    target: float = TARGET_COVERAGE,
) -> float:
    """Smallest multiplier whose coverage reaches the target.

    Smallest rather than closest: over-wide intervals are safe but useless, and
    picking the minimum that clears the bar keeps the bands as tight as honesty
    allows - which matters because band width drives safety stock.
    """
    best, best_err = 1.0, float("inf")
    for m in SEARCH:
        cov, n = coverage(mu, sigma, y_lower, event, target, m)
        if n == 0 or not np.isfinite(cov):
            continue
        if cov >= target:
            return float(m)
        err = abs(cov - target)
        if err < best_err:
            best, best_err = float(m), err
    return best


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate the TTM interval width")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--report", action="store_true", help="report the curve without refitting")
    args = ap.parse_args()

    t0 = time.perf_counter()
    booster = xgb.Booster()
    booster.load_model(str(ARTIFACTS / "survival_xgb.json"))
    meta = json.loads((ARTIFACTS / "survival_xgb_meta.json").read_text(encoding="utf-8"))
    feats = meta["features"]

    snaps = pd.read_parquet(args.data / "train_snapshots.parquet")
    feats = [f for f in feats if f in snaps.columns]

    # The calibration assets were held out of fitting by survival_xgb.train.
    cal_assets = set(meta["calibration_assets"])
    cal = snaps[(snaps["fold"] == "train") & (snaps["asset_id"].isin(cal_assets))]
    test = snaps[snaps["fold"] == "test"]
    print(f"calibration rows {len(cal):,} (held-out assets)  |  test rows {len(test):,}")

    def predict(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mu = booster.predict(to_dmatrix(df, feats), output_margin=True)
        return (
            mu,
            sigma_for(df, meta),
            df["y_lower"].to_numpy(dtype=float),
            df["event"].to_numpy(dtype=bool),
        )

    cal_mu, cal_sig, cal_y, cal_ev = predict(cal)
    te_mu, te_sig, te_y, te_ev = predict(test)

    multiplier = float(meta.get("calibration_multiplier", 1.0))
    by_type: dict[str, float] = dict(meta.get("calibration_multiplier_by_type", {}))

    if not args.report:
        multiplier = fit_multiplier(cal_mu, cal_sig, cal_y, cal_ev)

        # Per-type multipliers on top of the global one. A single scalar cannot
        # fix per-type miscalibration: measured at the global multiplier,
        # contactors covered 0.69 while TXVs covered 0.94, so one number is
        # simultaneously too tight for some parts and too loose for others.
        # Shrunk toward the global value by effective sample size so a type with
        # 40 calibration rows does not get a confidently wrong width.
        cal_t = cal.copy()
        cal_t["_mu"], cal_t["_sig"] = cal_mu, cal_sig
        for ctype, g in cal_t.groupby("component_type"):
            _, n_used = coverage(
                g["_mu"].to_numpy(), g["_sig"].to_numpy(),
                g["y_lower"].to_numpy(dtype=float), g["event"].to_numpy(dtype=bool),
                TARGET_COVERAGE, multiplier,
            )
            if n_used < 60:
                continue
            m_type = fit_multiplier(
                g["_mu"].to_numpy(), g["_sig"].to_numpy(),
                g["y_lower"].to_numpy(dtype=float), g["event"].to_numpy(dtype=bool),
            )
            w = n_used / (n_used + 250.0)
            by_type[str(ctype)] = float(w * m_type + (1.0 - w) * multiplier)

        meta["calibration_multiplier"] = multiplier
        meta["calibration_multiplier_by_type"] = by_type
        meta["calibration_target"] = TARGET_COVERAGE
        (ARTIFACTS / "survival_xgb_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def mult_for(df: pd.DataFrame) -> np.ndarray:
        return (
            df["component_type"].astype(str).map(lambda t: by_type.get(t, multiplier)).to_numpy(dtype=float)
        )

    pre, n_pre = coverage(cal_mu, cal_sig, cal_y, cal_ev, TARGET_COVERAGE, 1.0)
    post, _ = coverage(cal_mu, cal_sig, cal_y, cal_ev, TARGET_COVERAGE, multiplier)
    print(
        f"\nwidth multiplier s = {multiplier:.2f}"
        f"\n  calibration-set coverage of the nominal {TARGET_COVERAGE:.0%} band:"
        f"  {pre:.3f} -> {post:.3f}   (n={n_pre:,})"
    )

    # The calibration plot for the Models tab: nominal vs empirical, measured on
    # the *test* split so it is not the data the multiplier was fitted on.
    te_mult = mult_for(test)
    rows = []
    for level in NOMINAL_LEVELS:
        raw, n = coverage(te_mu, te_sig, te_y, te_ev, level, 1.0)
        cald, _ = coverage(te_mu, te_sig * te_mult, te_y, te_ev, level, 1.0)
        rows.append(
            {"nominal": level, "empirical_raw": raw, "empirical_calibrated": cald, "n_rows": n}
        )
    curve = pd.DataFrame(rows)
    curve.to_parquet(ARTIFACTS / "calibration_curve.parquet", index=False)

    print("\ncalibration curve (held-out test split):")
    print(
        curve.assign(
            nominal=lambda d: (d["nominal"] * 100).round().astype(int).astype(str) + "%",
            empirical_raw=lambda d: d["empirical_raw"].round(3),
            empirical_calibrated=lambda d: d["empirical_calibrated"].round(3),
        ).to_string(index=False)
    )

    # Per-type coverage after calibration: where is the band still wrong?
    test = test.copy()
    test["_mu"], test["_sig"] = te_mu, te_sig
    per_type = []
    for ctype, g in test.groupby("component_type"):
        m = by_type.get(str(ctype), multiplier)
        cov, n = coverage(
            g["_mu"].to_numpy(), g["_sig"].to_numpy(),
            g["y_lower"].to_numpy(dtype=float), g["event"].to_numpy(dtype=bool),
            TARGET_COVERAGE, m,
        )
        if n >= 30:
            per_type.append(
                {"component_type": ctype, "coverage": round(cov, 3), "multiplier": round(m, 2), "n": n}
            )
    pt = pd.DataFrame(per_type).sort_values("coverage")
    pt.to_parquet(ARTIFACTS / "coverage_by_type.parquet", index=False)
    print(f"\nper-type coverage at {TARGET_COVERAGE:.0%} nominal (n>=30):")
    print(pt.to_string(index=False))
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
