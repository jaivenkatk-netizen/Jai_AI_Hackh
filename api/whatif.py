"""Single-asset what-if: the one place the model runs inside a request.

Every other endpoint is a SELECT against a precomputed table. This path
deliberately recomputes, because "drag ambient to +6 C and watch the urgent
bucket grow" is the clearest demonstration that this is a model and not a
dashboard over a spreadsheet.

It stays fast by staying narrow - one asset, not five hundred:

    twin(ambient + d, load + d)   ->  healthy baseline under the scenario
      ->  shifted condition features
      ->  XGBoost AFT re-score for this asset's components
      ->  new TTM bands and horizon buckets
      ->  Monte Carlo for this asset's parts demand

The scenario shifts operating *conditions*, never the model and never the
output. A heatwave raises the thermal-stress features the survival model was
trained on, so the TTM shortens for a physical reason. No slider is wired to a
prediction.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from contracts.rules import assign_bucket, p_fail_within, ttm_band
from contracts.schemas import Criticality
from sim.asset_twin import DISPLAY_SIGNALS, solve, spec_for_asset

ARTIFACTS = Path(__file__).resolve().parents[1] / "models" / "artifacts"
MAX_TTM_DAYS = 3650.0

# Features a scenario is allowed to move. Everything else is held at its
# observed value, so the before/after difference isolates the condition change.
AMBIENT_LEVEL_FEATURES = ("ambient_t_max_mean_30d", "ambient_t_max_mean_90d")
DEGREE_DAY_FEATURES = {"degree_days_30d": 30.0, "degree_days_90d": 90.0}
LOAD_LEVEL_FEATURES = ("load_frac_mean_30d", "load_frac_mean_90d")
DUTY_LEVEL_FEATURES = ("runtime_h_mean_30d", "runtime_h_mean_90d")
DESIGN_AMBIENT_C = 35.0

# Scenario presets the UI offers as buttons. Named so the panel can label the
# comparison without the front end inventing physics.
PRESETS: dict[str, dict[str, float]] = {
    "baseline": {"d_ambient": 0.0, "d_load": 0.0, "d_duty": 0.0},
    "heatwave": {"d_ambient": 6.0, "d_load": 0.08, "d_duty": 2.0},
    "mild_week": {"d_ambient": -4.0, "d_load": -0.08, "d_duty": -1.5},
    "peak_occupancy": {"d_ambient": 0.0, "d_load": 0.15, "d_duty": 3.0},
    "night_setback": {"d_ambient": 0.0, "d_load": -0.2, "d_duty": -4.0},
}


class WhatIfEngine:
    """Holds the booster so a request does not pay model-load latency."""

    def __init__(self) -> None:
        self.booster = xgb.Booster()
        self.booster.load_model(str(ARTIFACTS / "survival_xgb.json"))
        # One thread: a 3.9 GB box, and the work is 20 rows wide
        self.booster.set_param({"nthread": 1})
        meta = json.loads((ARTIFACTS / "survival_xgb_meta.json").read_text(encoding="utf-8"))
        self.features: list[str] = meta["features"]
        self.sigma_by_type: dict[str, float] = meta["sigma_by_type"]
        self.sigma_global = float(meta["sigma_global"])
        self.mult_global = float(meta.get("calibration_multiplier", 1.0))
        self.mult_by_type: dict[str, float] = meta.get("calibration_multiplier_by_type", {})
        # First predict on a fresh booster costs ~2.4 s in thread-pool and
        # buffer setup. Pay it at startup so the first slider drag is as fast
        # as every one after it.
        warm = np.zeros((1, len(self.features)), dtype=np.float32)
        self.booster.predict(
            xgb.DMatrix(warm, feature_names=self.features), output_margin=True
        )
        # CoolProp loads its fluid library on the first property call, which is
        # the larger share of that cold-start cost. Run one throwaway solve so
        # the library is resident before a request needs it.
        try:
            solve(spec_for_asset("chiller", 280.0, "R32"), 30.0, 0.55)
        except Exception:
            pass  # the twin is optional to the re-score; don't block startup

    def _sigma(self, ctype: str) -> tuple[float, float]:
        return (
            float(self.sigma_by_type.get(ctype, self.sigma_global)),
            float(self.mult_by_type.get(ctype, self.mult_global)),
        )

    def predict(self, rows: pd.DataFrame) -> list[dict]:
        """Score one asset's components and return bands + buckets."""
        x = rows.reindex(columns=self.features).to_numpy(dtype=np.float32)
        mu = self.booster.predict(
            xgb.DMatrix(x, feature_names=self.features, missing=np.nan), output_margin=True
        )
        out = []
        for i, r in enumerate(rows.itertuples(index=False)):
            ctype = str(r.component_type)
            sigma, mult = self._sigma(ctype)
            band = ttm_band(float(mu[i]), sigma, mult)
            crit = Criticality(str(getattr(r, "criticality", "B")))
            p14 = p_fail_within(float(mu[i]), sigma, 14, mult)
            out.append(
                {
                    "component_id": str(r.component_id),
                    "component_type": ctype,
                    "criticality": crit.value,
                    "p10_days": round(min(band.p10_days, MAX_TTM_DAYS), 2),
                    "p50_days": round(min(band.p50_days, MAX_TTM_DAYS), 2),
                    "p90_days": round(min(band.p90_days, MAX_TTM_DAYS), 2),
                    "p_fail_14d": round(p14, 4),
                    "p_fail_90d": round(p_fail_within(float(mu[i]), sigma, 90, mult), 4),
                    "bucket": assign_bucket(p14, band.p50_days, crit).value,
                }
            )
        return out


def apply_scenario(
    rows: pd.DataFrame, d_ambient: float, d_load: float, d_duty: float
) -> pd.DataFrame:
    """Shift the condition features a scenario is allowed to touch.

    Degree-days are recomputed rather than scaled, because they are a clipped
    quantity: +3 C on a mild month adds almost nothing while +3 C on a hot one
    adds a great deal. Scaling would smear the effect evenly and understate the
    heatwave exactly where it matters.

    Slopes are left alone. A sustained shift in conditions changes the level,
    not the trend, and moving both would double-count the scenario.
    """
    out = rows.copy()

    for c in AMBIENT_LEVEL_FEATURES:
        if c in out.columns:
            out[c] = out[c].to_numpy(dtype=float) + d_ambient

    amb30 = rows["ambient_t_max_mean_30d"].to_numpy(dtype=float) if (
        "ambient_t_max_mean_30d" in rows.columns
    ) else None
    if amb30 is not None:
        for c, window in DEGREE_DAY_FEATURES.items():
            if c not in out.columns:
                continue
            before = np.clip(amb30 - DESIGN_AMBIENT_C, 0.0, None)
            after = np.clip(amb30 + d_ambient - DESIGN_AMBIENT_C, 0.0, None)
            out[c] = out[c].to_numpy(dtype=float) + (after - before) * window

    for c in LOAD_LEVEL_FEATURES:
        if c in out.columns:
            out[c] = np.clip(out[c].to_numpy(dtype=float) + d_load, 0.0, 1.0)
    for c in DUTY_LEVEL_FEATURES:
        if c in out.columns:
            out[c] = np.clip(out[c].to_numpy(dtype=float) + d_duty, 0.0, 24.0)

    # head_pressure_per_ambient is a ratio of two things the scenario moves;
    # recompute it so it stays consistent with its own numerator.
    if {"head_pressure_per_ambient", "discharge_p_mean_30d"} <= set(out.columns) and amb30 is not None:
        denom = np.clip(amb30 + d_ambient, 1.0, None)
        out["head_pressure_per_ambient"] = out["discharge_p_mean_30d"].to_numpy(dtype=float) / denom

    return out


def twin_view(
    asset: pd.Series, observed: dict[str, float], ambient_c: float, load_frac: float
) -> dict:
    """Twin state under these conditions, plus the residual against telemetry.

    This is what lets the panel say *why* the TTM moved: the head pressure a
    healthy machine of this design would show at this ambient, and how far the
    real machine already sits from it.
    """
    spec = spec_for_asset(
        str(asset["asset_type"]), float(asset["capacity_kw"]), str(asset["refrigerant"])
    )
    s = solve(spec, ambient_c, float(np.clip(load_frac, 0.1, 1.0)))
    base = {
        "discharge_p": s.discharge_p_kpa,
        "suction_p": s.suction_p_kpa,
        "superheat": s.superheat_k,
        "subcool": s.subcool_k,
        "comp_current": s.comp_current_a,
        "cop": s.cop,
    }
    signals = {}
    for k, b in base.items():
        v = observed.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        signals[k] = {
            "observed": round(float(v), 2),
            "healthy": round(float(b), 2),
            "residual": round(float(v) - float(b), 2),
        }
    return {
        "ambient_c": round(ambient_c, 1),
        "load_frac": round(float(np.clip(load_frac, 0.1, 1.0)), 3),
        "t_evap_c": round(s.t_evap_c, 2),
        "t_cond_c": round(s.t_cond_c, 2),
        "suction_p_kpa": round(s.suction_p_kpa, 1),
        "discharge_p_kpa": round(s.discharge_p_kpa, 1),
        "superheat_k": round(s.superheat_k, 2),
        "subcool_k": round(s.subcool_k, 2),
        "comp_current_a": round(s.comp_current_a, 1),
        "cop": round(s.cop, 3),
        "capacity_kw": round(s.q_evap_w / 1000.0, 1),
        "pressure_ratio": round(s.pressure_ratio, 3),
        # Below the unloading floor the machine cycles: the gauges read the
        # same running values and only the duty changes, which is why a load
        # increase on an already lightly-loaded machine moves the efficiency
        # without moving the pressures.
        "duty_frac": round(s.duty_frac, 3),
        "cycling": bool(s.duty_frac < 0.999),
        "head_pressure_limited": bool(s.head_pressure_limited),
        "cop_compressor": round(s.cop_compressor, 3),
        "converged": bool(s.converged),
        "signals": signals,
    }


def parts_delta(
    before: list[dict],
    after: list[dict],
    kits: pd.DataFrame,
    mix: pd.DataFrame,
    size_class: str,
    refrigerant: str,
    draws: int = 600,
    seed: int = 5,
) -> list[dict]:
    """Monte Carlo this asset's urgent parts demand, before and after.

    The same machinery as `planning/parts_demand.py`, but one asset wide so it
    runs inside a request. Both legs use the same seeded generator, so a
    difference in the result is a difference in the distribution, not noise.
    """
    kit_idx: dict[tuple, list[tuple[str, float, float]]] = {}
    for k in kits.itertuples(index=False):
        if str(k.size_class) != size_class or str(k.refrigerant) != refrigerant:
            continue
        kit_idx.setdefault((str(k.component_type), str(k.failure_mode)), []).append(
            (str(k.part_no), float(k.p_used), float(k.qty_mean))
        )
    mode_idx: dict[str, tuple[list[str], np.ndarray]] = {}
    for ctype, g in mix.groupby("component_type"):
        p = g["p_mode"].to_numpy(dtype=float)
        if p.sum() <= 0:
            continue
        mode_idx[str(ctype)] = (g["failure_mode"].astype(str).tolist(), p / p.sum())

    def run(preds: list[dict]) -> dict[str, float]:
        rng = np.random.default_rng(seed)  # same stream for both legs
        acc: dict[str, np.ndarray] = {}
        for pr in preds:
            ctype = pr["component_type"]
            if ctype not in mode_idx:
                continue
            # Rebuild the sampling distribution from the reported band, so the
            # parts figure can never disagree with the band shown beside it.
            p50 = max(pr["p50_days"], 0.5)
            p90 = max(pr["p90_days"], p50 * 1.01)
            mu = math.log(p50)
            sigma = max((math.log(p90) - mu) / 1.2815515655446004, 1e-3)
            urgent = np.exp(rng.normal(mu, sigma, draws)) <= 14.0
            modes, probs = mode_idx[ctype]
            drawn = rng.integers(0, len(modes), draws) if len(modes) == 1 else rng.choice(
                len(modes), size=draws, p=probs
            )
            roll = rng.random(draws)
            if not urgent.any():
                continue
            for mi, mode in enumerate(modes):
                for part_no, p_used, qty in kit_idx.get((ctype, mode), []):
                    used = urgent & (drawn == mi) & (roll < p_used)
                    if used.any():
                        acc.setdefault(part_no, np.zeros(draws))[:] += used * qty
        return {k: float(np.percentile(v, 90)) for k, v in acc.items()}

    b, a = run(before), run(after)
    rows = []
    for part in sorted(set(b) | set(a)):
        bv, av = b.get(part, 0.0), a.get(part, 0.0)
        if bv == 0.0 and av == 0.0:
            continue
        rows.append(
            {
                "part_no": part,
                "p90_before": round(bv, 2),
                "p90_after": round(av, 2),
                "delta": round(av - bv, 2),
            }
        )
    return sorted(rows, key=lambda r: (-abs(r["delta"]), r["part_no"]))


def _scenario_label(d_ambient: float, d_load: float, d_duty: float) -> str:
    bits = []
    if d_ambient:
        bits.append(f"ambient {d_ambient:+.1f} C")
    if d_load:
        bits.append(f"load {d_load:+.0%}")
    if d_duty:
        bits.append(f"duty {d_duty:+.1f} h/day")
    return ", ".join(bits) if bits else "baseline (no change)"


def run_whatif(
    con,
    engine: WhatIfEngine,
    asset_id: str,
    d_ambient: float = 0.0,
    d_load: float = 0.0,
    d_duty: float = 0.0,
) -> dict:
    """Re-score one asset under shifted conditions. Raises KeyError if unknown."""
    t0 = time.perf_counter()

    asset_df = con.execute("SELECT * FROM assets WHERE asset_id = ?", [asset_id]).df()
    if asset_df.empty:
        raise KeyError(asset_id)
    asset = asset_df.iloc[0]

    rows = con.execute("SELECT * FROM score_snapshots WHERE asset_id = ?", [asset_id]).df()
    if rows.empty:
        raise KeyError(f"no scored components for {asset_id}")

    before = engine.predict(rows)
    after = engine.predict(apply_scenario(rows, d_ambient, d_load, d_duty))

    cols = ", ".join(DISPLAY_SIGNALS)
    tel = con.execute(
        f"SELECT {cols}, ambient_t_max, load_frac FROM telemetry_daily "
        "WHERE asset_id = ? ORDER BY ts DESC LIMIT 1",
        [asset_id],
    ).df()
    if tel.empty:
        observed, amb_now, load_now = {}, 35.0, 0.8
    else:
        t = tel.iloc[0]
        observed = {k: float(t[k]) for k in DISPLAY_SIGNALS if pd.notna(t[k])}
        amb_now = float(t["ambient_t_max"])
        load_now = float(t["load_frac"])

    kits = con.execute("SELECT * FROM repair_kits").df()
    mix = con.execute("SELECT * FROM failure_mode_mix").df()
    deltas = parts_delta(
        before, after, kits, mix, str(asset["size_class"]), str(asset["refrigerant"])
    )

    # Bucket migration is the headline the panel shows
    was_of = {b["component_id"]: b["bucket"] for b in before}
    migration: dict[str, int] = {}
    moved = []
    for a in after:
        was, now = was_of.get(a["component_id"]), a["bucket"]
        if was and was != now:
            migration[f"{was} -> {now}"] = migration.get(f"{was} -> {now}", 0) + 1
            moved.append(a["component_id"])

    def counts(preds: list[dict]) -> dict[str, int]:
        c = {"urgent": 0, "planned": 0, "forecast": 0}
        for p in preds:
            c[p["bucket"]] = c.get(p["bucket"], 0) + 1
        return c

    return {
        "asset_id": asset_id,
        "asset_label": str(asset["asset_label"]),
        "asset_type": str(asset["asset_type"]),
        "site_name": str(asset["site_name"]),
        "hub_id": str(asset["hub_id"]),
        "scenario_label": _scenario_label(d_ambient, d_load, d_duty),
        "d_ambient": d_ambient,
        "d_load": d_load,
        "d_duty": d_duty,
        "twin_before": twin_view(asset, observed, amb_now, load_now),
        "twin_after": twin_view(asset, observed, amb_now + d_ambient, load_now + d_load),
        "components_before": before,
        "components_after": after,
        "bucket_counts_before": counts(before),
        "bucket_counts_after": counts(after),
        "bucket_migration": migration,
        "moved_component_ids": moved,
        "parts_delta": deltas,
        "assumptions": [
            "The scenario shifts operating conditions only - the trained model, "
            "its calibration and the component's accumulated wear are unchanged.",
            "Ambient and load are held at the shifted level for the whole 30 and "
            "90 day feature window, so this is a sustained shift, not one hot day.",
            "Degree-days above the 35 C design point are recomputed rather than "
            "scaled, so a shift on an already-hot site bites harder than on a mild one.",
            "Trend features (slopes) are left unchanged: a sustained shift moves "
            "the level, not the rate of change.",
            "Parts figures are the P90 of a 600-draw Monte Carlo over the 0-14 day "
            "horizon for this asset alone, and exclude scheduled PM.",
            "The survival model is a gradient-boosted tree ensemble with no "
            "monotonicity constraint, so a single component's TTM can tick the "
            "wrong way across a small condition step even though the trend over "
            "the full range is correct. Measured on a 0 to +12 C ambient sweep: "
            "43 of 144 half-degree steps moved against the trend, the largest by "
            "6.4%. Reading the direction over several degrees is reliable; reading "
            "one small step is not.",
        ],
        "compute_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }
