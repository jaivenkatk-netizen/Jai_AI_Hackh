"""Service needs -> parts and quantities, as a distribution.

This is the join the brief asks for and the step most systems get wrong by
taking expectations too early. The chain is:

    TTM distribution per component   (models/survival_xgb + conformal)
      -> sampled failure times       (1,000 draws per component)
      -> horizon bucket per draw     (urgent / planned / forecast)
      -> failure mode per draw       (P(mode | component_type) from history)
      -> parts per draw              (Bernoulli on P(used) from repair_kits)
      -> demand distribution per (part, hub, bucket)

Keeping it probabilistic to the end is the whole point. The mean is reported
for reference but never used to stock: a decision-aware benchmark of 38
intermittent-demand methods found the most *accurate* forecasters routinely
under-forecast and miss service targets, because minimising squared error and
meeting a 98% fill rate are different objectives. The P90 is what the inventory
policy consumes.

Double counting is avoided explicitly. Filters and belts are replaced both on
failure and at scheduled PM visits, so a component is charged to condition-
driven demand only if its predicted failure lands *before* its next PM;
otherwise it is charged to the deterministic PM schedule.

Run:
    python -m planning.parts_demand
    python -m planning.parts_demand --draws 2000 --widen-band 1.5
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.rules import PLANNED_MAX_DAYS, URGENT_MAX_DAYS
from sim.catalog import COMPONENTS

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"

N_DRAWS = 1000
BUCKETS = ("urgent", "planned", "forecast")
PM_INTERVAL_DAYS = 91
FORECAST_HORIZON_DAYS = 365  # beyond this, a draw contributes to no bucket


def _bucket_index(ttm: np.ndarray) -> np.ndarray:
    """0=urgent, 1=planned, 2=forecast, 3=beyond horizon (ignored)."""
    return np.digitize(ttm, [URGENT_MAX_DAYS, PLANNED_MAX_DAYS, FORECAST_HORIZON_DAYS])


def next_pm_due(work_orders: pd.DataFrame, assets: pd.DataFrame) -> dict[str, float]:
    """Days from now until each asset's next scheduled PM visit."""
    pm = work_orders[work_orders["type"] == "PM"]
    as_of = pd.Timestamp(work_orders["open_ts"].max())
    last = pm.groupby("asset_id")["open_ts"].max()
    out: dict[str, float] = {}
    for aid in assets["asset_id"]:
        lp = last.get(aid)
        if lp is None or pd.isna(lp):
            out[str(aid)] = PM_INTERVAL_DAYS / 2.0
        else:
            elapsed = (as_of - pd.Timestamp(lp)).days
            due = PM_INTERVAL_DAYS - (elapsed % PM_INTERVAL_DAYS)
            out[str(aid)] = float(max(due, 1))
    return out


def simulate_demand(
    ttm: pd.DataFrame,
    kits: pd.DataFrame,
    mix: pd.DataFrame,
    assets: pd.DataFrame,
    pm_due: dict[str, float],
    draws: int = N_DRAWS,
    widen_band: float = 1.0,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monte Carlo the parts demand. Returns (demand rows, contribution rows).

    `widen_band` multiplies every sigma - the lever used to prove that
    uncertainty actually propagates to safety stock rather than being decorative.
    """
    rng = np.random.default_rng(seed)

    amap = assets.set_index("asset_id")[["hub_id", "size_class", "refrigerant", "criticality"]]
    ttm = ttm.merge(amap, left_on="asset_id", right_index=True, how="left")

    # Repair kits indexed for fast lookup: (ctype, mode, size, refrig) -> parts
    kit_idx: dict[tuple, list[tuple[str, float, float]]] = defaultdict(list)
    for k in kits.itertuples(index=False):
        kit_idx[(k.component_type, k.failure_mode, k.size_class, k.refrigerant)].append(
            (k.part_no, float(k.p_used), float(k.qty_mean))
        )

    # Failure-mode mix per component type
    mode_idx: dict[str, tuple[list[str], np.ndarray]] = {}
    for ctype, g in mix.groupby("component_type"):
        p = g["p_mode"].to_numpy(dtype=float)
        mode_idx[str(ctype)] = (g["failure_mode"].tolist(), p / p.sum())

    # Accumulator: (part, hub) -> (3 buckets, draws). ~1k keys x 3 x draws is a
    # few MB, which matters on a 3.9GB box - accumulating per-component part
    # matrices instead would be two orders of magnitude larger.
    acc: dict[tuple[str, str], np.ndarray] = {}
    contrib: list[dict] = []

    for row in ttm.itertuples(index=False):
        ctype = str(row.component_type)
        hub = str(row.hub_id)
        if hub == "nan" or ctype not in mode_idx:
            continue

        sigma = float(row.sigma_log) * float(row.calibration_multiplier) * widen_band
        samples = np.exp(rng.normal(float(row.mu_log_days), max(sigma, 1e-9), draws))

        # Avoid double counting with the PM schedule: a consumable predicted to
        # outlast its next PM visit will be replaced at that visit instead, and
        # is accounted for in the deterministic PM demand below.
        pm_replaceable = COMPONENTS[ctype].pm_replaceable if ctype in COMPONENTS else False
        if pm_replaceable:
            due = pm_due.get(str(row.asset_id), PM_INTERVAL_DAYS)
            samples = np.where(samples <= due, samples, np.inf)

        bidx = _bucket_index(samples)
        modes, mprobs = mode_idx[ctype]
        drawn_modes = rng.choice(len(modes), size=draws, p=mprobs)

        for mi, mode in enumerate(modes):
            parts = kit_idx.get((ctype, mode, row.size_class, row.refrigerant))
            if not parts:
                continue
            active = drawn_modes == mi
            if not active.any():
                continue
            for part_no, p_used, qty in parts:
                used = active & (rng.random(draws) < p_used)
                if not used.any():
                    continue
                key = (part_no, hub)
                arr = acc.get(key)
                if arr is None:
                    arr = np.zeros((len(BUCKETS), draws), dtype=np.float32)
                    acc[key] = arr
                for b in range(len(BUCKETS)):
                    sel = used & (bidx == b)
                    if sel.any():
                        arr[b] += sel * qty
                # Traceability: which asset/mode fed this part's urgent demand
                urgent_share = float((used & (bidx == 0)).mean())
                if urgent_share > 0.02:
                    contrib.append(
                        {
                            "part_no": part_no,
                            "hub_id": hub,
                            "bucket": BUCKETS[0],
                            "asset_id": row.asset_id,
                            "component_id": row.component_id,
                            "component_type": ctype,
                            "failure_mode": mode,
                            "p_contribution": urgent_share,
                        }
                    )

    rows = []
    for (part_no, hub), arr in acc.items():
        for b, bucket in enumerate(BUCKETS):
            v = arr[b]
            if v.max() <= 0:
                continue
            rows.append(
                {
                    "part_no": part_no,
                    "hub_id": hub,
                    "bucket": bucket,
                    "demand_p10": float(np.percentile(v, 10)),
                    "demand_p50": float(np.percentile(v, 50)),
                    "demand_p90": float(np.percentile(v, 90)),
                    "demand_mean": float(v.mean()),
                    "demand_sd": float(v.std()),
                    "draws": draws,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(contrib)


def pm_scheduled_demand(
    ttm: pd.DataFrame,
    kits: pd.DataFrame,
    assets: pd.DataFrame,
    pm_due: dict[str, float],
) -> pd.DataFrame:
    """Deterministic consumable demand from the PM calendar.

    Filters and belts get changed on schedule whether or not they have failed,
    so this is a known quantity rather than a distribution - and it is usually
    the largest single line in a spares budget.
    """
    amap = assets.set_index("asset_id")[["hub_id", "size_class", "refrigerant"]]
    ttm = ttm.merge(amap, left_on="asset_id", right_index=True, how="left")
    kit_idx: dict[tuple, list[tuple[str, float]]] = defaultdict(list)
    for k in kits.itertuples(index=False):
        kit_idx[(k.component_type, k.failure_mode, k.size_class, k.refrigerant)].append(
            (k.part_no, float(k.qty_mean))
        )

    rows: list[dict] = []
    for row in ttm.itertuples(index=False):
        ctype = str(row.component_type)
        if ctype not in COMPONENTS or not COMPONENTS[ctype].pm_replaceable:
            continue
        due = pm_due.get(str(row.asset_id), PM_INTERVAL_DAYS)
        # Only count the component if it is expected to survive to its PM visit
        if float(row.p50_days) <= due:
            continue
        modes = COMPONENTS[ctype].failure_modes
        mode = modes[0][0]
        for part_no, qty in kit_idx.get((ctype, mode, row.size_class, row.refrigerant), []):
            bucket = (
                "urgent" if due <= URGENT_MAX_DAYS
                else ("planned" if due <= PLANNED_MAX_DAYS else "forecast")
            )
            rows.append(
                {"part_no": part_no, "hub_id": row.hub_id, "bucket": bucket, "qty": qty}
            )
    if not rows:
        return pd.DataFrame(columns=["part_no", "hub_id", "bucket", "pm_demand"])
    df = pd.DataFrame(rows)
    return (
        df.groupby(["part_no", "hub_id", "bucket"], as_index=False)["qty"]
        .sum()
        .rename(columns={"qty": "pm_demand"})
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Monte Carlo parts demand per part x hub x horizon")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--widen-band",
        type=float,
        default=1.0,
        help="multiply every sigma; used to show uncertainty propagating to stock",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    ttm = pd.read_parquet(args.data / "component_ttm.parquet")
    kits = pd.read_parquet(args.data / "repair_kits.parquet")
    mix = pd.read_parquet(args.data / "failure_mode_mix.parquet")
    assets = pd.read_parquet(args.data / "assets.parquet")
    work_orders = pd.read_parquet(args.data / "work_orders.parquet")
    parts = pd.read_parquet(args.data / "parts.parquet")

    pm_due = next_pm_due(work_orders, assets)
    demand, contrib = simulate_demand(
        ttm, kits, mix, assets, pm_due,
        draws=args.draws, widen_band=args.widen_band, seed=args.seed,
    )
    pm_dem = pm_scheduled_demand(ttm, kits, assets, pm_due)

    demand = demand.merge(pm_dem, on=["part_no", "hub_id", "bucket"], how="outer")
    for c in ("demand_p10", "demand_p50", "demand_p90", "demand_mean", "demand_sd"):
        demand[c] = demand[c].fillna(0.0)
    demand["pm_demand"] = demand["pm_demand"].fillna(0.0)
    demand["draws"] = demand["draws"].fillna(args.draws).astype(int)
    # Total planning quantity: the probabilistic P90 plus the known PM schedule
    demand["total_p50"] = demand["demand_p50"] + demand["pm_demand"]
    demand["total_p90"] = demand["demand_p90"] + demand["pm_demand"]

    demand = demand.merge(
        parts[["part_no", "description", "unit_cost", "criticality"]], on="part_no", how="left"
    )
    suffix = "" if args.widen_band == 1.0 else f"_widen{args.widen_band:g}"
    demand.to_parquet(args.data / f"parts_demand{suffix}.parquet", index=False)
    contrib.to_parquet(args.data / f"parts_demand_contrib{suffix}.parquet", index=False)

    print(f"{len(demand):,} (part, hub, bucket) rows; {len(contrib):,} traceable contributions")
    print("\ndemand by bucket:")
    print(
        demand.groupby("bucket")
        .agg(rows=("part_no", "size"), p50=("total_p50", "sum"), p90=("total_p90", "sum"))
        .round(1)
        .to_string()
    )
    print("\ntop urgent demand lines (P90, next 14 days):")
    u = demand[demand["bucket"] == "urgent"].nlargest(10, "total_p90")
    print(
        u[["part_no", "description", "hub_id", "demand_p50", "demand_p90", "pm_demand",
           "criticality", "unit_cost"]].round(2).to_string(index=False)
    )
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
