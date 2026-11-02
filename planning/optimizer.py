"""Shortages -> approvable actions: transfer, buy, expedite or defer.

The daily decision. For every (part, hub) projected short before its need date,
compare four options and pick the cheapest one that still meets the date:

  transfer  a nearby hub holds stock above its *own* P90 need, and the transfer
            lands in time. Cheapest real option, and it is checked first - the
            answer to a shortage should be redeployment before procurement.
  purchase  normal lead time fits before the need date.
  expedite  need date falls inside the lead time and no hub can donate.
  defer     low-criticality part where the risk of waiting is small.

A greedy pass is enough here and is what ships: shortages are processed in need-
date order, and donor stock is decremented as it is committed so two hubs cannot
be promised the same unit. A daily MIP (minimise holding + transfer + expedite +
criticality-weighted stockout) is the stronger version and is listed as a next
step rather than half-built.

Every card carries the three things the brief requires - the action, the
confidence, and the assumptions - and `evidence` is assembled from the actual
numbers behind the decision so a planner can audit it without leaving the
screen.

Run:
    python -m planning.optimizer
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"

# Cost model. Transfers carry a handling charge plus a small share of unit value
# (packing, insurance); expediting carries the supplier's premium.
TRANSFER_FIXED_COST = 450.0
TRANSFER_VALUE_RATE = 0.02
HOLDING_RATE_PER_DAY = 0.00045  # ~16%/yr carrying cost
# Days an MOQ overshoot is assumed to sit on the shelf before being consumed.
PLANNING_HORIZON_FOR_MOQ = 90.0
# Downtime cost per day of waiting, by criticality - what a deferral risks.
DOWNTIME_COST_PER_DAY = {"A": 42_000.0, "B": 11_000.0, "C": 2_200.0}
# Probability-weighting on that downtime: a deferred job does not certainly
# fail during the wait, it fails with roughly the band's probability mass.
DEFER_RISK_WEIGHT = 0.25
DEFER_MAX_CRITICALITY = "C"


def need_dates(contrib: pd.DataFrame, ttm: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Earliest credible need date per (part, hub), from contributing components.

    Uses each contributing component's P10 - the optimistic end of its band -
    because planning to the median means arriving late half the time on the
    very components that drive the urgent queue.
    """
    if contrib.empty:
        return pd.DataFrame(columns=["part_no", "hub_id", "need_in_days", "need_by", "drivers"])
    j = contrib.merge(
        ttm[["component_id", "p10_days", "p50_days", "p_fail_14d", "confidence"]],
        on="component_id",
        how="left",
    )
    rows = []
    for (part_no, hub), g in j.groupby(["part_no", "hub_id"]):
        g = g.sort_values("p10_days")
        soonest = float(g["p10_days"].min())
        rows.append(
            {
                "part_no": part_no,
                "hub_id": hub,
                "need_in_days": soonest,
                "need_by": as_of + pd.Timedelta(days=float(max(soonest, 1))),
                "n_drivers": int(g["component_id"].nunique()),
                "drivers": json.dumps(
                    [
                        {
                            "asset_id": r.asset_id,
                            "component_id": r.component_id,
                            "component_type": r.component_type,
                            "failure_mode": r.failure_mode,
                            "p_fail_14d": None if pd.isna(r.p_fail_14d) else round(float(r.p_fail_14d), 3),
                            "p50_days": None if pd.isna(r.p50_days) else round(float(r.p50_days), 1),
                            "confidence": r.confidence,
                        }
                        for r in g.head(5).itertuples(index=False)
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_recommendations(
    positions: pd.DataFrame,
    needs: pd.DataFrame,
    transfers: pd.DataFrame,
    suppliers: pd.DataFrame,
    hubs: pd.DataFrame,
    ttm: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Greedy transfer-vs-buy over all projected shortages."""
    pos = positions.merge(needs, on=["part_no", "hub_id"], how="left")
    pos["need_in_days"] = pos["need_in_days"].fillna(90.0)

    # Donor pool: stock a hub holds above its own P90 requirement. Decremented
    # as transfers are committed so the same unit is never promised twice.
    donor: dict[tuple[str, str], float] = {
        (r.part_no, r.hub_id): float(r.excess_units) for r in positions.itertuples(index=False)
    }
    tmap: dict[tuple[str, str], float] = {
        (t.from_hub_id, t.to_hub_id): float(t.transfer_days) for t in transfers.itertuples(index=False)
    }
    sup = suppliers.set_index("part_no")
    hub_name = hubs.set_index("hub_id")["name"].to_dict()

    short = pos[pos["shortfall_vs_p90"] > 0].sort_values(["need_in_days", "criticality"])
    cards: list[dict] = []
    rec_i = 0

    for s in short.itertuples(index=False):
        qty = float(math.ceil(s.shortfall_vs_p90))
        if qty <= 0:
            continue
        part_no, hub = str(s.part_no), str(s.hub_id)
        crit = str(s.criticality)
        need_days = float(s.need_in_days)
        unit_cost = float(s.unit_cost)

        srow = sup.loc[part_no] if part_no in sup.index else None
        lt_mean = float(srow["lead_time_mean_d"]) if srow is not None else 10.0
        lt_sd = float(srow["lead_time_sd_d"]) if srow is not None else 2.0
        moq = int(srow["moq"]) if srow is not None else 1
        expedite_cost = float(srow["expedite_cost"]) if srow is not None else unit_cost * 0.9

        # Costs below are *incremental*, excluding the part's own value.
        #
        # Comparing total outlays is wrong and produced nonsense: deferring a
        # 58-unit contactor shortage looked optimal because "spend 85,840 on
        # parts" lost to "risk 3,630 of downtime". But the parts are needed
        # either way - buying them converts cash into an asset rather than
        # consuming it. What actually differs between the options is the
        # *premium* each one pays to get the part there in time: transfer
        # handling, carrying cost, an expedite fee, or accepted downtime risk.
        # Comparing premiums reproduces what a planner does, and makes deferral
        # win only when the risk really is smaller than the cheapest premium.

        # --- option 1: lateral transfer -------------------------------------
        best_donor, best_days, best_avail = None, None, 0.0
        for (dpart, dhub), avail in donor.items():
            if dpart != part_no or dhub == hub or avail < 1:
                continue
            days = tmap.get((dhub, hub))
            if days is None or days > need_days:
                continue
            if best_days is None or days < best_days:
                best_donor, best_days, best_avail = dhub, days, avail

        transfer_qty = min(qty, float(np.floor(best_avail))) if best_donor else 0.0
        transfer_cost = (
            TRANSFER_FIXED_COST + TRANSFER_VALUE_RATE * unit_cost * transfer_qty
            if transfer_qty >= 1
            else math.inf
        )

        # --- option 2: normal purchase order --------------------------------
        # Premium is the carrying cost of holding it over the lead time, plus
        # the carrying cost of any MOQ overshoot we did not actually need.
        po_fits = lt_mean <= need_days
        po_qty = max(qty, moq)
        moq_overshoot = max(po_qty - qty, 0.0)
        po_cost = (
            unit_cost * po_qty * HOLDING_RATE_PER_DAY * lt_mean
            + unit_cost * moq_overshoot * HOLDING_RATE_PER_DAY * PLANNING_HORIZON_FOR_MOQ
            if po_fits
            else math.inf
        )

        # --- option 3: expedite ---------------------------------------------
        exp_cost = expedite_cost * qty

        # --- option 4: defer -------------------------------------------------
        # Risk-weighted downtime from waiting out the lead time. Restricted to
        # low-criticality parts: deferring a class-A failure is a business
        # decision, not an optimisation.
        wait_days = max(lt_mean - need_days, 0.0)
        defer_cost = (
            DOWNTIME_COST_PER_DAY[crit] * wait_days * DEFER_RISK_WEIGHT
            if crit == DEFER_MAX_CRITICALITY
            else math.inf
        )

        options = {
            "transfer": transfer_cost,
            "purchase": po_cost,
            "expedite": exp_cost,
            "defer": defer_cost,
        }
        action = min(options, key=lambda k: options[k])
        if not math.isfinite(options[action]):
            action, options["expedite"] = "expedite", exp_cost

        chosen_qty = {
            "transfer": transfer_qty, "purchase": po_qty, "expedite": max(qty, moq), "defer": 0.0
        }[action]
        if action == "transfer" and best_donor:
            donor[(part_no, best_donor)] = max(best_avail - chosen_qty, 0.0)

        # Cheapest alternative, for "what this saves"
        alt = sorted((v, k) for k, v in options.items() if k != action and math.isfinite(v))
        avoided = alt[0][1] if alt else None
        cost_delta = (options[action] - alt[0][0]) if alt else 0.0

        # --- the card's prose ------------------------------------------------
        drivers = json.loads(s.drivers) if isinstance(s.drivers, str) else []
        n_drivers = int(s.n_drivers) if not pd.isna(getattr(s, "n_drivers", np.nan)) else len(drivers)
        hub_label = hub_name.get(hub, hub)
        donor_label = hub_name.get(best_donor, best_donor) if best_donor else None

        if action == "transfer":
            why = (
                f"{n_drivers} predicted repair(s) at {hub_label} need {part_no} within "
                f"{need_days:.0f} days; only {s.available:.0f} on hand against a P90 need of "
                f"{s.projected_need_p90:.0f}. Supplier lead time of {lt_mean:.0f} days misses the "
                f"date, but {donor_label} holds {best_avail:.0f} above its own P90 requirement and "
                f"a transfer lands in {best_days:.0f} days - cheaper than expediting."
            )
        elif action == "purchase":
            moq_note = (
                f" Ordering {chosen_qty:.0f} rather than {qty:.0f} to meet the supplier minimum."
                if chosen_qty > qty
                else ""
            )
            why = (
                f"{hub_label} is short {qty:.0f} of {part_no} against a P90 need of "
                f"{s.projected_need_p90:.0f} within {need_days:.0f} days. Supplier lead time of "
                f"{lt_mean:.0f} days fits before the need date, so a normal purchase order is the "
                f"cheapest option that meets it.{moq_note}"
            )
        elif action == "expedite":
            moq_note = (
                f" Ordering {chosen_qty:.0f} to meet the supplier minimum."
                if chosen_qty > qty
                else ""
            )
            why = (
                f"{hub_label} needs {qty:.0f} of {part_no} within {need_days:.0f} days, inside the "
                f"{lt_mean:.0f}-day lead time, and no hub holds surplus above its own P90 that can "
                f"arrive in time. Expediting is the only option that meets the date for a class-"
                f"{crit} part.{moq_note}"
            )
        else:
            why = (
                f"{hub_label} is short {qty:.0f} of {part_no}, but it is a class-{crit} part and "
                f"the risk of waiting {wait_days:.0f} days for normal replenishment "
                f"(~{defer_cost:,.0f} risk-weighted) is below the cost of expediting "
                f"(~{exp_cost:,.0f}), so the job is better rescheduled."
            )

        evidence = [
            f"On hand {s.available:.0f}, reorder point {s.reorder_point:.1f}, "
            f"safety stock {s.safety_stock:.1f} at {s.service_level_target:.0%} service level "
            f"(class {crit})",
            f"P50 need {s.projected_need_p50:.1f}, P90 need {s.projected_need_p90:.1f} over 90 days",
            f"Supplier lead time {lt_mean:.1f} d (sd {lt_sd:.1f}) vs {need_days:.0f} d to need date",
        ]
        for dv in drivers[:3]:
            evidence.append(
                f"{dv['asset_id']} {dv['component_type']} / {dv['failure_mode']}: "
                f"P(fail <= 14 d) = {dv['p_fail_14d']}, P50 {dv['p50_days']} d"
            )
        if action == "transfer" and best_donor:
            evidence.append(
                f"{donor_label} holds {best_avail:.0f} above its own P90 need; transfer "
                f"{best_days:.0f} d"
            )

        # Confidence inherits the weakest contributing prediction - a transfer
        # justified by three low-confidence forecasts is not a high-confidence
        # recommendation, however tidy the arithmetic.
        levels = [dv.get("confidence") for dv in drivers if dv.get("confidence")]
        order = {"high": 0, "medium": 1, "low": 2}
        conf = max(levels, key=lambda x: order.get(x, 1)) if levels else "medium"

        assumptions = [
            {"label": "Lead time", "text": f"supplier mean {lt_mean:.0f} d holds",
             "basis": "supplier history"},
            {"label": "Duty cycle", "text": "continues at the current rate",
             "basis": "trailing 30-day mean"},
        ]
        if action == "transfer":
            assumptions.append(
                {"label": "Transfer time", "text": f"takes {best_days:.0f} days",
                 "basis": "hub distance matrix"}
            )
        if conf != "high":
            assumptions.append(
                {"label": "Forecast confidence",
                 "text": f"weakest contributing prediction is {conf} confidence",
                 "basis": "component confidence badges"}
            )

        rec_i += 1
        cards.append(
            {
                "rec_id": f"REC-{rec_i:04d}",
                "action": action,
                "part_no": part_no,
                "description": s.description if hasattr(s, "description") else part_no,
                "qty": chosen_qty,
                "from_hub_id": best_donor if action == "transfer" else None,
                "to_hub_id": hub,
                "need_by": as_of + pd.Timedelta(days=float(max(need_days, 1))),
                "need_in_days": need_days,
                "criticality": crit,
                "why": why,
                "evidence": json.dumps(evidence),
                "confidence": conf,
                "assumptions": json.dumps(assumptions),
                "cost_delta": float(cost_delta),
                "action_cost": float(options[action]),
                "avoided_action": avoided,
                "unit_cost": unit_cost,
                "status": "pending",
            }
        )

    return pd.DataFrame(cards)


def main() -> None:
    ap = argparse.ArgumentParser(description="Greedy transfer-vs-buy recommendations")
    ap.add_argument("--data", type=Path, default=DATA)
    args = ap.parse_args()

    t0 = time.perf_counter()
    d = args.data
    positions = pd.read_parquet(d / "stock_positions.parquet")
    contrib = pd.read_parquet(d / "parts_demand_contrib.parquet")
    ttm = pd.read_parquet(d / "component_ttm.parquet")
    transfers = pd.read_parquet(d / "hub_transfers.parquet")
    suppliers = pd.read_parquet(d / "suppliers.parquet")
    hubs = pd.read_parquet(d / "hubs.parquet")
    parts = pd.read_parquet(d / "parts.parquet")
    as_of = pd.Timestamp(pd.read_parquet(d / "work_orders.parquet", columns=["open_ts"])["open_ts"].max())

    positions = positions.merge(parts[["part_no", "description"]], on="part_no", how="left")
    needs = need_dates(contrib, ttm, as_of)
    cards = build_recommendations(positions, needs, transfers, suppliers, hubs, ttm, as_of)
    cards.to_parquet(d / "recommendations.parquet", index=False)

    print(f"{len(cards):,} recommendation cards")
    if cards.empty:
        print("no shortages found")
        return
    print("\nby action:")
    print(
        cards.groupby("action")
        .agg(n=("rec_id", "size"), qty=("qty", "sum"), cost=("action_cost", "sum"))
        .round(0).to_string()
    )
    print("\nexpedites avoided by transferring instead: "
          f"{int((cards['avoided_action'] == 'expedite').sum())}")
    print("\nmost urgent cards:")
    show = cards.nsmallest(5, "need_in_days")
    for c in show.itertuples(index=False):
        print(f"\n  [{c.rec_id}] {c.action.upper()} {c.qty:.0f} x {c.part_no} -> {c.to_hub_id}"
              f"  (need in {c.need_in_days:.0f} d, confidence {c.confidence})")
        print(f"    {c.why}")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
