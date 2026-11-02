"""Reorder points and safety stock, driven by the TTM forecast.

What makes this different from a classic min/max planner: demand during the
lead time comes from the condition-based forecast, not from past consumption.
A hub whose assets are degrading now gets a higher reorder point than an
identical hub whose assets are healthy, even if their histories match.

Method. For each (part, hub) the quantity that matters is demand *over the
replenishment lead time*, and both demand and lead time are random. Rather than
scale a fixed-horizon forecast by L/90 - which quietly assumes demand arrives
uniformly and that the two distributions are independent of each other - the
lead time is sampled per draw and the component failures are counted inside
that sampled window:

    L ~ LogNormal(supplier mean, sd)
    D_L = sum over components of  1[TTM <= L] x 1[part used] x qty
    ROP = quantile(D_L, service level)      SS = ROP - E[D_L]

Service level comes from the part's criticality class (A 98% / B 95% / C 90%),
which is how "reflect service criticality" enters the arithmetic.

The closed form  SS = z * sqrt(L*sd_D^2 + D^2*sd_L^2)  is reported alongside for
explainability - it is what the recommendation card cites - but the numbers the
system acts on are the Monte Carlo ones, which need no normality assumption.

Run:
    python -m planning.inventory_policy
    python -m planning.inventory_policy --sensitivity --widen-band 1.5
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.rules import safety_stock_closed_form
from contracts.schemas import Criticality
from planning.parts_demand import PM_INTERVAL_DAYS, next_pm_due
from sim.catalog import COMPONENTS

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"

N_DRAWS = 1000
PLANNING_HORIZON_DAYS = 90  # window for "projected need" and excess flags


def lognormal_params(mean: float, sd: float) -> tuple[float, float]:
    """Mu/sigma of a lognormal with the given mean and sd (method of moments)."""
    mean = max(float(mean), 0.5)
    sd = max(float(sd), 1e-3)
    var = sd**2
    sigma2 = np.log1p(var / mean**2)
    return float(np.log(mean) - sigma2 / 2.0), float(np.sqrt(sigma2))


def sample_lead_times(
    suppliers: pd.DataFrame, draws: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Lead-time draws per part, with occasional supplier lateness baked in."""
    out: dict[str, np.ndarray] = {}
    for s in suppliers.itertuples(index=False):
        mu, sigma = lognormal_params(s.lead_time_mean_d, s.lead_time_sd_d)
        L = rng.lognormal(mu, sigma, draws)
        # A chronically late supplier slips badly on a fraction of orders; the
        # on-time rate is observed in the supplier table, so model it rather
        # than pretending every order lands on the mean.
        late = rng.random(draws) > float(getattr(s, "on_time_rate", 0.95))
        L = np.where(late, L * rng.uniform(1.4, 2.2, draws), L)
        out[str(s.part_no)] = L
    return out


def simulate_policy(
    ttm: pd.DataFrame,
    kits: pd.DataFrame,
    mix: pd.DataFrame,
    assets: pd.DataFrame,
    parts: pd.DataFrame,
    lead_samples: dict[str, np.ndarray],
    pm_due: dict[str, float],
    draws: int = N_DRAWS,
    widen_band: float = 1.0,
    seed: int = 11,
) -> pd.DataFrame:
    """Demand over lead time, and demand over the planning horizon, per part x hub."""
    rng = np.random.default_rng(seed)

    amap = assets.set_index("asset_id")[["hub_id", "size_class", "refrigerant"]]
    ttm = ttm.merge(amap, left_on="asset_id", right_index=True, how="left")

    kit_idx: dict[tuple, list[tuple[str, float, float]]] = defaultdict(list)
    for k in kits.itertuples(index=False):
        kit_idx[(k.component_type, k.failure_mode, k.size_class, k.refrigerant)].append(
            (k.part_no, float(k.p_used), float(k.qty_mean))
        )
    mode_idx: dict[str, tuple[list[str], np.ndarray]] = {}
    for ctype, g in mix.groupby("component_type"):
        p = g["p_mode"].to_numpy(dtype=float)
        mode_idx[str(ctype)] = (g["failure_mode"].tolist(), p / p.sum())

    # (part, hub) -> [demand over lead time, demand over horizon], each (draws,)
    acc_lead: dict[tuple[str, str], np.ndarray] = {}
    acc_horizon: dict[tuple[str, str], np.ndarray] = {}

    for row in ttm.itertuples(index=False):
        ctype = str(row.component_type)
        hub = str(row.hub_id)
        if hub == "nan" or ctype not in mode_idx:
            continue

        sigma = float(row.sigma_log) * float(row.calibration_multiplier) * widen_band
        samples = np.exp(rng.normal(float(row.mu_log_days), max(sigma, 1e-9), draws))
        if ctype in COMPONENTS and COMPONENTS[ctype].pm_replaceable:
            due = pm_due.get(str(row.asset_id), PM_INTERVAL_DAYS)
            samples = np.where(samples <= due, samples, np.inf)

        modes, mprobs = mode_idx[ctype]
        drawn = rng.choice(len(modes), size=draws, p=mprobs)

        for mi, mode in enumerate(modes):
            for part_no, p_used, qty in kit_idx.get(
                (ctype, mode, row.size_class, row.refrigerant), []
            ):
                active = (drawn == mi) & (rng.random(draws) < p_used)
                if not active.any():
                    continue
                L = lead_samples.get(part_no)
                key = (part_no, hub)
                if L is not None:
                    need_in_lead = active & (samples <= L)
                    if need_in_lead.any():
                        arr = acc_lead.get(key)
                        if arr is None:
                            arr = np.zeros(draws, dtype=np.float32)
                            acc_lead[key] = arr
                        arr += need_in_lead * qty
                need_in_horizon = active & (samples <= PLANNING_HORIZON_DAYS)
                if need_in_horizon.any():
                    arr = acc_horizon.get(key)
                    if arr is None:
                        arr = np.zeros(draws, dtype=np.float32)
                        acc_horizon[key] = arr
                    arr += need_in_horizon * qty

    crit = parts.set_index("part_no")["criticality"].to_dict()
    cost = parts.set_index("part_no")["unit_cost"].to_dict()

    rows = []
    for key in set(acc_lead) | set(acc_horizon):
        part_no, hub = key
        dl = acc_lead.get(key, np.zeros(draws, dtype=np.float32))
        dh = acc_horizon.get(key, np.zeros(draws, dtype=np.float32))
        c = Criticality(str(crit.get(part_no, "C")))
        level = {"A": 0.98, "B": 0.95, "C": 0.90}[c.value]
        L = lead_samples.get(part_no, np.array([7.0]))

        rop = float(np.quantile(dl, level))
        mean_dl = float(dl.mean())
        rows.append(
            {
                "part_no": part_no,
                "hub_id": hub,
                "criticality": c.value,
                "service_level_target": level,
                "lead_time_mean_d": float(np.mean(L)),
                "lead_time_sd_d": float(np.std(L)),
                "demand_over_lead_mean": mean_dl,
                "demand_over_lead_sd": float(dl.std()),
                "reorder_point": rop,
                "safety_stock": max(rop - mean_dl, 0.0),
                "projected_need_p50": float(np.quantile(dh, 0.50)),
                "projected_need_p90": float(np.quantile(dh, 0.90)),
                "unit_cost": float(cost.get(part_no, 0.0)),
                # The textbook formula, for the card's explanation line only.
                "safety_stock_closed_form": safety_stock_closed_form(
                    mean_demand_per_day=mean_dl / max(float(np.mean(L)), 1.0),
                    sd_demand_per_day=float(dl.std()) / max(np.sqrt(float(np.mean(L))), 1.0),
                    lead_time_mean_d=float(np.mean(L)),
                    lead_time_sd_d=float(np.std(L)),
                    criticality=c,
                ),
            }
        )
    return pd.DataFrame(rows)


def add_positions(policy: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Join on-hand stock and flag shortages and excess."""
    pos = policy.merge(inventory, on=["part_no", "hub_id"], how="left")
    for c in ("on_hand", "on_order", "reserved"):
        pos[c] = pos[c].fillna(0)
    pos["available"] = pos["on_hand"] + pos["on_order"] - pos["reserved"]
    pos["shortfall_vs_rop"] = (pos["reorder_point"] - pos["available"]).clip(lower=0)
    pos["shortfall_vs_p90"] = (pos["projected_need_p90"] - pos["available"]).clip(lower=0)
    # Excess: stock above the 90-day P90 need. Flagged before any new buy is
    # suggested anywhere, so the first answer to a shortage is redeployment.
    pos["excess_units"] = (pos["available"] - pos["projected_need_p90"]).clip(lower=0)
    pos["excess_value"] = pos["excess_units"] * pos["unit_cost"]
    daily = pos["projected_need_p50"] / PLANNING_HORIZON_DAYS
    pos["days_of_cover"] = np.where(daily > 0, pos["available"] / daily, np.inf)
    return pos


def main() -> None:
    ap = argparse.ArgumentParser(description="Reorder point and safety stock per part x hub")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--widen-band", type=float, default=1.0)
    ap.add_argument(
        "--sensitivity",
        action="store_true",
        help="rerun at the widened band and report the change in class-A safety stock",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    d = args.data
    ttm = pd.read_parquet(d / "component_ttm.parquet")
    kits = pd.read_parquet(d / "repair_kits.parquet")
    mix = pd.read_parquet(d / "failure_mode_mix.parquet")
    assets = pd.read_parquet(d / "assets.parquet")
    parts = pd.read_parquet(d / "parts.parquet")
    suppliers = pd.read_parquet(d / "suppliers.parquet")
    inventory = pd.read_parquet(d / "inventory.parquet")
    work_orders = pd.read_parquet(d / "work_orders.parquet")
    pm_due = next_pm_due(work_orders, assets)

    rng = np.random.default_rng(3)
    lead = sample_lead_times(suppliers, args.draws, rng)

    def run(widen: float) -> pd.DataFrame:
        pol = simulate_policy(
            ttm, kits, mix, assets, parts, lead, pm_due,
            draws=args.draws, widen_band=widen,
        )
        return add_positions(pol, inventory)

    base = run(args.widen_band)
    suffix = "" if args.widen_band == 1.0 else f"_widen{args.widen_band:g}"
    base.to_parquet(d / f"stock_positions{suffix}.parquet", index=False)

    print(f"{len(base):,} (part, hub) positions")
    print("\nby criticality class:")
    print(
        base.groupby("criticality")
        .agg(
            positions=("part_no", "size"),
            mean_rop=("reorder_point", "mean"),
            mean_safety_stock=("safety_stock", "mean"),
            shortages=("shortfall_vs_rop", lambda s: int((s > 0).sum())),
            excess_value=("excess_value", "sum"),
        )
        .round(2)
        .to_string()
    )

    short = base[base["shortfall_vs_p90"] > 0].nlargest(10, "shortfall_vs_p90")
    print("\nlargest shortfalls against 90-day P90 need:")
    print(
        short[["part_no", "hub_id", "criticality", "available", "projected_need_p90",
               "shortfall_vs_p90", "reorder_point", "lead_time_mean_d", "unit_cost"]]
        .round(1).to_string(index=False)
    )
    print(f"\ntotal excess value: {base['excess_value'].sum():,.0f}")

    if args.sensitivity:
        wide = run(args.widen_band * 1.5 if args.widen_band != 1.0 else 1.5)
        j = base.merge(wide, on=["part_no", "hub_id"], suffixes=("", "_wide"))
        print("\n--- uncertainty propagation check ---")
        print(
            "If a wider TTM band does not raise safety stock, the core claim of the\n"
            "system is broken. Safety stock by criticality, base vs widened sigma:"
        )
        summary = j.groupby("criticality").agg(
            base_ss=("safety_stock", "mean"),
            wide_ss=("safety_stock_wide", "mean"),
            base_rop=("reorder_point", "mean"),
            wide_rop=("reorder_point_wide", "mean"),
        )
        summary["ss_change_pct"] = (
            (summary["wide_ss"] - summary["base_ss"]) / summary["base_ss"].replace(0, np.nan) * 100
        )
        print(summary.round(2).to_string())
        a = summary.loc["A"] if "A" in summary.index else None
        if a is not None:
            verdict = "PASS" if a["wide_ss"] > a["base_ss"] else "FAIL"
            print(f"\n  class-A safety stock {a['base_ss']:.2f} -> {a['wide_ss']:.2f}   {verdict}")

    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
