"""Assumption cards and confidence badges in plain words.

The brief requires predictions to "show confidence and assumptions". A band and
a badge answer *how sure*; this module answers *under what conditions*, which
is the part that lets a planner disagree with the model intelligently - if the
card says "assumes duty continues at 14 h/day" and the planner knows the
building is shutting for a fortnight, they know to re-run the what-if rather
than distrust the number wholesale.

Cards are templates filled from the actual feature values behind the
prediction, so they are specific and checkable rather than boilerplate. The
copilot can rephrase them, but it never invents them.

Run:
    python -m models.assumptions
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.rules import confidence_badge, p_fail_within, ttm_band

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# Beyond this, a TTM number is not a plan input - it is extrapolation past the
# observation window. Raw AFT output reached 56,584 days (155 years) for
# components showing no degradation, which is mathematically consistent with
# heavy censoring and useless on a screen. Capped values are flagged so the UI
# can render "beyond planning horizon" instead of a fake precise number; the
# Forecast bucket treats everything past 90 days identically anyway.
MAX_TTM_DAYS = 3650.0

CLIMATE_LABEL = {
    "1A": "hot-humid (zone 1A)",
    "1B": "hot-dry (zone 1B)",
    "2A": "warm-humid (zone 2A)",
    "2B": "warm-dry (zone 2B)",
}


def build_cards(row: pd.Series) -> list[dict[str, str | None]]:
    """Assumption cards for one component prediction."""
    cards: list[dict[str, str | None]] = []

    duty = row.get("runtime_h_mean_30d")
    if duty is not None and np.isfinite(duty):
        cards.append(
            {
                "label": "Duty cycle",
                "text": f"continues at about {duty:.0f} h/day",
                "basis": "trailing 30-day mean runtime",
            }
        )

    zone = row.get("climate_zone")
    amb = row.get("ambient_t_max_mean_30d")
    if zone and np.isfinite(amb if amb is not None else np.nan):
        cards.append(
            {
                "label": "Ambient",
                "text": (
                    f"follows the seasonal normal for {CLIMATE_LABEL.get(str(zone), str(zone))}; "
                    f"recent peak average {amb:.0f} C"
                ),
                "basis": "30-day observed peak ambient",
            }
        )

    dspm = row.get("days_since_pm")
    if dspm is not None and np.isfinite(dspm):
        cards.append(
            {
                "label": "Last preventive visit",
                "text": (
                    f"was {dspm:.0f} days ago and reset the consumable clocks"
                    if dspm < 400
                    else "is not on record inside the observation window"
                ),
                "basis": "work-order history",
            }
        )

    imputed = float(row.get("imputed_share_30d") or 0.0)
    gap_h = float(row.get("hours_since_reading") or 0.0)
    if imputed > 0.02 or gap_h > 24:
        bits = []
        if imputed > 0.02:
            bits.append(f"{imputed:.0%} of readings imputed")
        if gap_h > 24:
            bits.append(f"last reading {gap_h / 24:.0f} days old")
        cards.append(
            {
                "label": "Data quality",
                "text": " and ".join(bits),
                "basis": "telemetry completeness check (30 d)",
            }
        )
    else:
        cards.append(
            {
                "label": "Data quality",
                "text": "telemetry complete and current",
                "basis": "telemetry completeness check (30 d)",
            }
        )

    repairs = row.get("n_past_repairs")
    if repairs is not None and repairs > 0:
        cards.append(
            {
                "label": "Repair history",
                "text": (
                    f"{int(repairs)} previous repair{'s' if repairs > 1 else ''} on this component "
                    "are treated as full restorations"
                ),
                "basis": "as-good-as-new assumption after replacement",
            }
        )

    model_layer = row.get("model_layer", "survival_xgb")
    if model_layer == "weibull_prior":
        cards.append(
            {
                "label": "Model",
                "text": "no dedicated sensor on this part, so the fleet Weibull prior is used",
                "basis": "conditional on survival to current age",
            }
        )
    return cards


def enrich(preds: pd.DataFrame, score: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Attach bands, P(fail) horizons, badges and assumption cards."""
    mult_global = float(meta.get("calibration_multiplier", 1.0))
    mult_by_type = meta.get("calibration_multiplier_by_type", {})

    joined = preds.merge(
        score.drop(columns=[c for c in preds.columns if c in score.columns and c != "component_id"],
                   errors="ignore"),
        on="component_id",
        how="left",
        suffixes=("", "_s"),
    )

    # Self-calibrating "wide band" threshold: the fleet's upper-tercile relative
    # width. Computed in a first pass so the badge judges each band against what
    # this model actually achieves, rather than against a guessed constant.
    ratios = []
    for row in joined.itertuples(index=False):
        d = row._asdict()
        mult = float(mult_by_type.get(str(d.get("component_type", "")), mult_global))
        b = ttm_band(float(d["mu_log_days"]), float(d["sigma_log"]), mult)
        ratios.append(b.width_days / max(b.p50_days, 1e-6))
    wide_ratio = float(np.percentile(ratios, 66)) if ratios else 2.5

    out_rows = []
    for row in joined.itertuples(index=False):
        d = row._asdict()
        ctype = str(d.get("component_type", ""))
        mult = float(mult_by_type.get(ctype, mult_global))
        mu, sigma = float(d["mu_log_days"]), float(d["sigma_log"])

        band = ttm_band(mu, sigma, mult)
        s = pd.Series(d)
        badge = confidence_badge(
            band=band,
            sensor_coverage=1.0 - float(d.get("imputed_share_30d") or 0.0),
            hours_since_last_reading=float(d.get("hours_since_reading") or 0.0),
            imputed_share=float(d.get("imputed_share_30d") or 0.0),
            # Out-of-distribution proxy: a band wider than the fleet-typical
            # width is the cheapest available OOD signal. Clamped at 0 because a
            # *tighter*-than-average band means well-predicted, not unusual -
            # only excess width counts as evidence of being off-distribution.
            ood_score=max(0.0, float(min(sigma / max(meta["sigma_global"], 1e-6), 3.0)) - 1.0),
            wide_band_ratio=wide_ratio,
        )
        beyond = band.p50_days > MAX_TTM_DAYS
        out_rows.append(
            {
                "component_id": d["component_id"],
                "asset_id": d.get("asset_id"),
                "component_type": ctype,
                "mu_log_days": mu,
                "sigma_log": sigma,
                "calibration_multiplier": mult,
                "p10_days": min(band.p10_days, MAX_TTM_DAYS),
                "p50_days": min(band.p50_days, MAX_TTM_DAYS),
                "p90_days": min(band.p90_days, MAX_TTM_DAYS),
                "beyond_horizon": bool(beyond),
                "sigma_effective": band.sigma_log,
                "p_fail_14d": p_fail_within(mu, sigma, 14, mult),
                "p_fail_30d": p_fail_within(mu, sigma, 30, mult),
                "p_fail_90d": p_fail_within(mu, sigma, 90, mult),
                "p_fail_180d": p_fail_within(mu, sigma, 180, mult),
                "confidence": badge.level.value,
                "confidence_reasons": json.dumps(badge.reasons),
                "data_quality": badge.data_quality,
                "ood_score": badge.ood_score,
                # Capped width, so it stays consistent with the capped p10/p90
                # written above. Mixing an uncapped width with a capped median
                # made high-confidence components look like they had the widest
                # bands, which is the opposite of what the badge means.
                "interval_width_days": min(band.p90_days, MAX_TTM_DAYS)
                - min(band.p10_days, MAX_TTM_DAYS),
                "rel_width": badge.interval_width_days / max(band.p50_days, 1e-6),
                "assumptions": json.dumps(build_cards(s)),
            }
        )
    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build TTM bands, badges and assumption cards")
    ap.add_argument("--data", type=Path, default=DATA)
    args = ap.parse_args()

    t0 = time.perf_counter()
    meta = json.loads((ARTIFACTS / "survival_xgb_meta.json").read_text(encoding="utf-8"))
    preds = pd.read_parquet(args.data / "component_predictions.parquet")
    score = pd.read_parquet(args.data / "score_snapshots.parquet")

    enriched = enrich(preds, score, meta)
    enriched.to_parquet(args.data / "component_ttm.parquet", index=False)

    print(f"{len(enriched):,} components with bands, badges and assumptions")
    print(f"  beyond {MAX_TTM_DAYS:,.0f}-day planning horizon: "
          f"{int(enriched['beyond_horizon'].sum()):,}")
    print("\nconfidence mix:")
    print(enriched["confidence"].value_counts().to_string())
    # Relative width, not absolute: the badge judges a band against its own
    # median, because 20 days of uncertainty means something very different on
    # a 9-day prediction than on a 2000-day one.
    print("\nband width relative to its own median, by confidence (should rise H->L):")
    print(enriched.groupby("confidence")["rel_width"].median().round(2).to_string())
    print("\ndata quality by confidence (should fall H->L):")
    print(enriched.groupby("confidence")["data_quality"].median().round(2).to_string())

    urgent = enriched[enriched["p_fail_14d"] >= 0.30]
    print(f"\ncomponents with P(fail <= 14 d) >= 0.30 : {len(urgent):,}")
    print(
        enriched.nsmallest(6, "p50_days")[
            ["component_id", "component_type", "p10_days", "p50_days", "p90_days",
             "p_fail_14d", "confidence"]
        ].round(2).to_string(index=False)
    )
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
