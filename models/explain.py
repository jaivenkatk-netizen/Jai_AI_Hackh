"""SHAP drivers per component - why this part is degrading, not just that it is.

The brief asks predictions to carry their reasoning. SHAP on the AFT margin
gives exactly that: a signed contribution per feature in log-days, so a driver
can be stated as "pushes the predicted life down" with a magnitude attached.

Sign convention matters and is easy to invert. The model predicts mu = E[log
TTM], so a *positive* SHAP value lengthens predicted life and a negative one
shortens it. The UI wants "what is killing this component", so drivers are
ranked by |contribution| and labelled `shortens_life` when the SHAP value is
negative.

Feature names are turned into readable labels mechanically from the naming
convention (signal + statistic + window), so adding a feature upstream does not
require touching a hand-written dictionary here.

Run:
    python -m models.explain
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from models.survival_xgb import to_dmatrix

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
TOP_K = 5

SIGNAL_LABELS = {
    "suction_p": "Suction pressure",
    "discharge_p": "Discharge (head) pressure",
    "superheat": "Superheat",
    "subcool": "Subcooling",
    "supply_t": "Supply air temperature",
    "return_t": "Return air temperature",
    "comp_current": "Compressor current",
    "vib_rms": "Vibration RMS",
    "vib_kurtosis": "Vibration kurtosis",
    "current_imbalance": "Current imbalance",
    "cop": "COP (efficiency)",
    "charge_frac": "Refrigerant charge",
    "ambient_t_max": "Peak ambient temperature",
    "runtime_h": "Runtime hours",
    "load_frac": "Load fraction",
    "starts": "Compressor starts",
    "short_cycles": "Short-cycling events",
}

STAT_LABELS = {"mean": "average", "std": "variability", "slope": "trend"}

STATIC_LABELS = {
    "age_days": "Days since this part was fitted",
    "days_since_pm": "Days since last preventive visit",
    "n_past_repairs": "Previous repairs on this component",
    "asset_age_days": "Asset age",
    "capacity_kw": "Machine capacity",
    "nominal_duty_h": "Scheduled duty hours",
    "imputed_share_30d": "Share of readings imputed (30 d)",
    "hours_since_reading": "Hours since last telemetry",
    "head_pressure_per_ambient": "Head pressure relative to ambient",
    "delta_t_air": "Air-side temperature split",
    "degree_days_30d": "Degree-days above design (30 d)",
    "degree_days_90d": "Degree-days above design (90 d)",
    "starts_mean_7d": "Starts per day (7 d average)",
}


def humanise(feature: str) -> str:
    """Turn a feature name into something a planner can read."""
    if feature in STATIC_LABELS:
        return STATIC_LABELS[feature]
    for prefix, pretty in (
        ("component_type_", "Component type"),
        ("asset_type_", "Machine type"),
        ("size_class_", "Size class"),
        ("climate_zone_", "Climate zone"),
        ("criticality_", "Service criticality"),
    ):
        if feature.startswith(prefix):
            return f"{pretty}: {feature[len(prefix):].replace('_', ' ')}"
    parts = feature.rsplit("_", 2)
    if len(parts) == 3:
        sig, stat, window = parts
        if sig in SIGNAL_LABELS and stat in STAT_LABELS:
            return f"{SIGNAL_LABELS[sig]} {STAT_LABELS[stat]} ({window})"
    return feature.replace("_", " ")


def top_drivers(
    booster: xgb.Booster, score: pd.DataFrame, feats: list[str], top_k: int = TOP_K
) -> pd.DataFrame:
    """Per-component top-k SHAP drivers, long format."""
    dmat = to_dmatrix(score, feats)
    # pred_contribs returns one column per feature plus a trailing bias term.
    contribs = booster.predict(dmat, pred_contribs=True)
    shap = contribs[:, :-1]
    values = score[feats].to_numpy(dtype=float)

    order = np.argsort(-np.abs(shap), axis=1)[:, :top_k]
    rows = []
    comp_ids = score["component_id"].to_numpy()
    for i, cid in enumerate(comp_ids):
        for rank, j in enumerate(order[i]):
            s = float(shap[i, j])
            if abs(s) < 1e-6:
                continue
            v = values[i, j]
            rows.append(
                {
                    "component_id": cid,
                    "rank": rank + 1,
                    "feature": feats[j],
                    "display_name": humanise(feats[j]),
                    "shap_value": s,
                    "value": None if not np.isfinite(v) else float(v),
                    # mu is log-TTM, so a negative contribution shortens life
                    "direction": "shortens_life" if s < 0 else "extends_life",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute SHAP drivers for every component")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    t0 = time.perf_counter()
    booster = xgb.Booster()
    booster.load_model(str(ARTIFACTS / "survival_xgb.json"))
    meta = json.loads((ARTIFACTS / "survival_xgb_meta.json").read_text(encoding="utf-8"))

    score = pd.read_parquet(args.data / "score_snapshots.parquet")
    feats = [f for f in meta["features"] if f in score.columns]

    drivers = top_drivers(booster, score, feats, args.top_k)
    drivers.to_parquet(args.data / "component_drivers.parquet", index=False)
    print(f"{len(drivers):,} driver rows for {drivers['component_id'].nunique():,} components")

    # What the fleet is dying of overall - a useful sanity check that the model
    # is keying on degradation signals and not just on component identity.
    shortening = drivers[drivers["direction"] == "shortens_life"]
    top = (
        shortening.groupby("display_name")
        .agg(components=("component_id", "nunique"), mean_impact=("shap_value", "mean"))
        .sort_values("components", ascending=False)
        .head(12)
    )
    print("\nmost common life-shortening drivers across the fleet:")
    print(top.round(3).to_string())
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
