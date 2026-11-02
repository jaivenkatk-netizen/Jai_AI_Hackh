"""Fleet Weibull prior per component type.

Three jobs:

  1. Baseline to beat. The evaluation table needs an honest, non-trivial
     comparator; "the fleet average life for this part type, conditioned on how
     long this one has already survived" is exactly what a good maintenance
     planner does today without telemetry.
  2. Fallback. Filters, belts and bearings often have no dedicated sensor. The
     conditional Weibull still gives them a calibrated TTM band, so the parts
     forecast covers the whole fleet rather than only instrumented components.
  3. Sanity check on the conditional model. If XGBoost cannot beat this, the
     telemetry is not carrying signal and that is worth knowing early.

The conditioning is the part people get wrong. A component that has already
survived `a` days does not have the unconditional median life remaining - for
a wear-out distribution (k > 1) it has *less*. The quantile solves

    P(T > a + r | T > a) = 1 - p   =>   r_p = lambda * ((a/lambda)^k - ln(1-p))^(1/k) - a

which is what `conditional_quantile` implements.

Run:
    python -m models.weibull_prior
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import WeibullFitter

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


# --------------------------------------------------------------------------
# Deriving component lives from the work-order history
# --------------------------------------------------------------------------


def component_lives(
    components: pd.DataFrame,
    work_orders: pd.DataFrame,
    as_of: pd.Timestamp,
    window_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Complete and censored lives per component type.

    A component's history is install -> fail -> fail -> ... -> (still running).
    Each gap between consecutive failures is a complete life; the trailing gap
    from the last failure to `as_of` is right-censored.

    Left truncation matters here and is easy to get wrong. For a component that
    was never replaced inside the observation window we do not know its true
    install date - only when its *asset* was commissioned, which is not the
    same thing (parts get swapped before we start watching). Treating
    asset-commissioning -> now as an observed life inflates long-lived parts
    absurdly: doing exactly that produced a 42,563-day median for the
    refrigerant circuit, because every one of its lives looked like a decades-
    long survivor.

    So `window_start` restricts the fit to lives whose *start* we actually
    observed. That discards left-truncated first lives rather than pretending
    to know when they began, which is the standard treatment and leaves plenty
    of data (the in-window failures all qualify).
    """
    cm = work_orders[(work_orders["type"] != "PM") & work_orders["component_id"].notna()]
    fails_by_comp = {cid: np.sort(g["open_ts"].to_numpy()) for cid, g in cm.groupby("component_id")}
    ws = np.datetime64(window_start) if window_start is not None else None

    rows = []
    n_truncated = 0
    for comp in components.itertuples(index=False):
        fails = fails_by_comp.get(comp.component_id, np.array([], dtype="datetime64[ns]"))
        prev = np.datetime64(pd.Timestamp(comp.install_date))
        observed_start = ws is None or prev >= ws
        for f in fails:
            dur = float((f - prev) / np.timedelta64(1, "D"))
            if dur > 0.5:
                if observed_start:
                    rows.append(
                        {"component_type": comp.component_type, "duration_days": dur, "event": True}
                    )
                else:
                    n_truncated += 1
            prev = f
            observed_start = True  # every subsequent life starts at a seen replacement
        tail = float((np.datetime64(as_of) - prev) / np.timedelta64(1, "D"))
        if tail > 0.5:
            if observed_start:
                rows.append(
                    {"component_type": comp.component_type, "duration_days": tail, "event": False}
                )
            else:
                n_truncated += 1
    out = pd.DataFrame(rows)
    out.attrs["n_left_truncated_dropped"] = n_truncated
    return out


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def fit_priors(lives: pd.DataFrame, min_events: int = 8) -> dict[str, dict[str, float]]:
    """Fit a Weibull per component type; pool the rest into a global prior."""
    priors: dict[str, dict[str, float]] = {}
    for ctype, g in lives.groupby("component_type"):
        n_events = int(g["event"].sum())
        if n_events < min_events:
            continue
        wf = WeibullFitter()
        wf.fit(g["duration_days"].to_numpy(), event_observed=g["event"].to_numpy())
        priors[ctype] = {
            "lambda": float(wf.lambda_),
            "k": float(wf.rho_),
            "n_lives": int(len(g)),
            "n_events": n_events,
            "median_life_days": float(wf.median_survival_time_),
        }

    # Global fallback for types with too few observed failures to fit alone.
    wf = WeibullFitter()
    wf.fit(lives["duration_days"].to_numpy(), event_observed=lives["event"].to_numpy())
    priors["__global__"] = {
        "lambda": float(wf.lambda_),
        "k": float(wf.rho_),
        "n_lives": int(len(lives)),
        "n_events": int(lives["event"].sum()),
        "median_life_days": float(wf.median_survival_time_),
    }
    return priors


# --------------------------------------------------------------------------
# Conditional prediction
# --------------------------------------------------------------------------


def conditional_quantile(age_days: np.ndarray, lam: float, k: float, p: float) -> np.ndarray:
    """Remaining life at failure-probability p, given survival to `age_days`.

    r_p = lambda * ((age/lambda)^k - ln(1 - p))^(1/k) - age
    """
    age = np.clip(np.asarray(age_days, dtype=float), 0.0, None)
    base = (age / lam) ** k
    total = lam * np.power(base - np.log(1.0 - p), 1.0 / k)
    return np.clip(total - age, 0.1, None)


def conditional_p_fail(age_days: np.ndarray, horizon_days: float, lam: float, k: float) -> np.ndarray:
    """P(fail within horizon | survived to age)."""
    age = np.clip(np.asarray(age_days, dtype=float), 0.0, None)
    s_now = np.exp(-((age / lam) ** k))
    s_then = np.exp(-(((age + horizon_days) / lam) ** k))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(1.0 - np.where(s_now > 0, s_then / s_now, 0.0), 0.0, 1.0)


def predict(
    df: pd.DataFrame, priors: dict[str, dict[str, float]], age_col: str = "age_days"
) -> pd.DataFrame:
    """P10/P50/P90 remaining life per row, using the matching type's prior."""
    out = pd.DataFrame(index=df.index)
    glob = priors["__global__"]
    p10 = np.empty(len(df))
    p50 = np.empty(len(df))
    p90 = np.empty(len(df))
    used = []
    for i, (ctype, age) in enumerate(zip(df["component_type"].to_numpy(), df[age_col].to_numpy())):
        pr = priors.get(str(ctype), glob)
        lam, k = pr["lambda"], pr["k"]
        p10[i] = conditional_quantile(np.array([age]), lam, k, 0.10)[0]
        p50[i] = conditional_quantile(np.array([age]), lam, k, 0.50)[0]
        p90[i] = conditional_quantile(np.array([age]), lam, k, 0.90)[0]
        used.append(str(ctype) if str(ctype) in priors else "__global__")
    out["prior_p10_days"] = p10
    out["prior_p50_days"] = p50
    out["prior_p90_days"] = p90
    out["prior_source"] = used
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit the fleet Weibull prior per component type")
    ap.add_argument("--data", type=Path, default=DATA)
    args = ap.parse_args()

    t0 = time.perf_counter()
    components = pd.read_parquet(args.data / "components.parquet")
    work_orders = pd.read_parquet(args.data / "work_orders.parquet")
    ts = pd.read_parquet(args.data / "telemetry_daily.parquet", columns=["ts"])["ts"]
    as_of, window_start = pd.Timestamp(ts.max()), pd.Timestamp(ts.min())

    lives = component_lives(components, work_orders, as_of, window_start=window_start)
    print(f"{len(lives):,} component lives with an observed start  "
          f"({int(lives['event'].sum()):,} failures, {1 - lives['event'].mean():.0%} censored); "
          f"dropped {lives.attrs.get('n_left_truncated_dropped', 0):,} left-truncated")

    priors = fit_priors(lives)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "weibull_priors.json").write_text(json.dumps(priors, indent=2), encoding="utf-8")

    rows = []
    for ctype, pr in sorted(priors.items(), key=lambda kv: kv[1]["median_life_days"]):
        rows.append(
            {
                "component_type": ctype,
                "k (shape)": round(pr["k"], 2),
                "lambda (scale, d)": round(pr["lambda"], 0),
                "median life (d)": round(pr["median_life_days"], 0),
                "lives": pr["n_lives"],
                "events": pr["n_events"],
            }
        )
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print(
        "\nk > 1 means wear-out (failure rate rises with age), which is what we expect\n"
        "for mechanical parts; k near 1 means roughly random failure."
    )
    print(f"\nwrote {ARTIFACTS / 'weibull_priors.json'}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
