"""Failure mode -> parts and quantities, learned from the work-order history.

This module deliberately does NOT import sim.catalog.REPAIR_KITS, even though
the generator's ground-truth kits are sitting right there. On real data nobody
hands you P(used); you recover it from what technicians actually consumed. So
the estimate comes from the simulated work orders exactly as it would from a
real CMMS export, and the generator's truth is used only to score the recovery
(see --validate), never to produce it.

What comes out:

  * P(used) per (component_type, failure_mode, part_no), Laplace-smoothed so a
    part seen on 1 of 2 repairs does not get a confident 50%.
  * Expected quantity per repair, conditional on the part being used.
  * A van-kit rule: parts above a probability threshold, plus cheap
    high-probability extras, which is the lever that moves first-time-fix.

Run:
    python -m planning.repair_kits
    python -m planning.repair_kits --validate   # compare against generator truth
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"

# Laplace / add-alpha smoothing. With alpha=1 a part seen on 1 of 2 repairs
# lands at (1+1)/(2+2)=0.50 but with far less leverage than an unsmoothed
# estimate once more evidence arrives - and a part seen 0 of 3 times keeps a
# small non-zero probability instead of being declared impossible.
ALPHA = 1.0

# Van-kit rule (see catalog.VAN_KIT_* for the generator-side equivalents).
VAN_KIT_MIN_P = 0.20
VAN_KIT_EXTRA_MAX_COST = 2_500.0
VAN_KIT_PRIMARY_P = 0.55


# Kits are conditioned on these asset properties, not just on the failure mode.
#
# Without them the estimate marginalises over the fleet: a compressor repair
# consumes CMP-S, CMP-M or CMP-L depending on machine size, so each part number
# recovers P(used) ~ 0.15-0.46 (summing to ~1.0) against a ground truth of 1.0
# for the size-appropriate one. Planning always knows the asset's size class and
# refrigerant, so conditioning on them recovers the right number and stops the
# demand forecast under-predicting by a factor of two.
VARIANT_KEYS = ["size_class", "refrigerant"]


def explode_work_orders(work_orders: pd.DataFrame) -> pd.DataFrame:
    """One row per (work order, part) from the list-valued parts columns."""
    cm = work_orders[(work_orders["type"] != "PM") & work_orders["failure_mode"].notna()].copy()
    rows = []
    for wo in cm.itertuples(index=False):
        # Parquet hands list columns back as numpy arrays, so `x or []` raises
        # "truth value of an array is ambiguous" - test for None explicitly.
        parts = [] if wo.parts_used is None else list(wo.parts_used)
        qtys = [] if wo.parts_qty is None else list(wo.parts_qty)
        for pn, q in zip(parts, qtys):
            rows.append(
                {
                    "wo_id": wo.wo_id,
                    "component_type": wo.component_type,
                    "failure_mode": wo.failure_mode,
                    "size_class": getattr(wo, "size_class", None),
                    "refrigerant": getattr(wo, "refrigerant", None),
                    "part_no": pn,
                    "qty": float(q),
                }
            )
    return pd.DataFrame(rows)


def derive_kits(
    work_orders: pd.DataFrame, assets: pd.DataFrame, min_repairs: int = 3
) -> pd.DataFrame:
    """Estimate P(used) and expected quantity per (mode, variant, part)."""
    wo = work_orders.merge(
        assets[["asset_id", *VARIANT_KEYS]], on="asset_id", how="left", suffixes=("", "_a")
    )
    cm = wo[(wo["type"] != "PM") & wo["failure_mode"].notna()]
    group = ["component_type", "failure_mode", *VARIANT_KEYS]

    # Denominator: how many repairs of this kind, on this kind of machine
    repairs = cm.groupby(group)["wo_id"].nunique().rename("n_repairs")

    used = explode_work_orders(wo)
    if used.empty:
        return pd.DataFrame()

    # Numerator: on how many of those repairs did this part appear
    per_part = used.groupby([*group, "part_no"]).agg(
        n_used=("wo_id", "nunique"),
        qty_mean=("qty", "mean"),
        qty_p90=("qty", lambda s: float(np.percentile(s, 90))),
    )

    kits = per_part.join(repairs, on=group).reset_index()
    kits = kits[kits["n_repairs"] >= min_repairs]
    kits["p_used"] = (kits["n_used"] + ALPHA) / (kits["n_repairs"] + 2.0 * ALPHA)
    # Expected units per repair = P(part is needed) x units when it is needed
    kits["expected_qty_per_repair"] = kits["p_used"] * kits["qty_mean"]
    return kits.sort_values([*group, "p_used"], ascending=[True, True, True, True, False])


def mode_mix(work_orders: pd.DataFrame) -> pd.DataFrame:
    """P(failure_mode | component_type) - which way a component tends to fail.

    Needed because the TTM model predicts *when* a component fails, not *how*.
    Parts demand has to marginalise over modes, since a compressor burnout and a
    mechanical wear repair pull different kits.
    """
    cm = work_orders[(work_orders["type"] != "PM") & work_orders["failure_mode"].notna()]
    counts = cm.groupby(["component_type", "failure_mode"])["wo_id"].nunique().rename("n")
    totals = counts.groupby("component_type").sum().rename("total")
    mix = counts.to_frame().join(totals, on="component_type").reset_index()
    mix["p_mode"] = mix["n"] / mix["total"]
    return mix


def build_van_kits(kits: pd.DataFrame, parts: pd.DataFrame) -> pd.DataFrame:
    """Which parts to put on the van for a given repair.

    Two tiers: parts likely enough to be the actual fix, and cheap parts worth
    carrying speculatively because a second visit costs far more than the part.
    """
    merged = kits.merge(parts[["part_no", "description", "unit_cost"]], on="part_no", how="left")
    merged = merged[merged["p_used"] >= VAN_KIT_MIN_P].copy()
    merged["reason"] = np.where(
        merged["p_used"] >= VAN_KIT_PRIMARY_P,
        "primary",
        np.where(
            merged["unit_cost"].fillna(1e9) <= VAN_KIT_EXTRA_MAX_COST,
            "cheap_high_probability_extra",
            "drop",
        ),
    )
    merged = merged[merged["reason"] != "drop"]
    # Expected first-time fix if every listed part is carried: the chance that
    # no *required* part is missing, approximated as the product over primaries.
    key = ["component_type", "failure_mode", *VARIANT_KEYS]
    ftf = (
        merged[merged["reason"] == "primary"]
        .groupby(key)["p_used"]
        .apply(lambda s: float(np.prod(np.clip(s, 0, 1)) ** 0.5))  # soft, not all-or-nothing
        .rename("expected_first_time_fix")
    )
    return merged.join(ftf, on=key)


def validate(kits: pd.DataFrame, bom: pd.DataFrame) -> pd.DataFrame:
    """Compare recovered P(used) against the generator's ground truth.

    Not used by the pipeline - this is the check that the recovery step works,
    which is the claim being made when we say the kits are learned from history.
    """
    truth = bom[
        ["component_type", "failure_mode", "size_class", "part_no", "p_used_truth"]
    ].drop_duplicates(subset=["component_type", "failure_mode", "size_class", "part_no"])
    cmp_df = kits.merge(
        truth, on=["component_type", "failure_mode", "size_class", "part_no"], how="inner"
    )
    if cmp_df.empty:
        return cmp_df
    cmp_df["abs_err"] = (cmp_df["p_used"] - cmp_df["p_used_truth"]).abs()
    return cmp_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive repair kits from work-order history")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    work_orders = pd.read_parquet(args.data / "work_orders.parquet")
    parts = pd.read_parquet(args.data / "parts.parquet")
    assets = pd.read_parquet(args.data / "assets.parquet")

    kits = derive_kits(work_orders, assets)
    mix = mode_mix(work_orders)
    van = build_van_kits(kits, parts)

    kits.to_parquet(args.data / "repair_kits.parquet", index=False)
    mix.to_parquet(args.data / "failure_mode_mix.parquet", index=False)
    van.to_parquet(args.data / "van_kits.parquet", index=False)

    print(f"{len(kits):,} (mode, variant, part) rows across "
          f"{kits.groupby(['component_type', 'failure_mode']).ngroups:,} failure modes")
    print(f"{len(van):,} van-kit rows")

    print("\nexample: compressor kit for an L-size R32 machine, recovered from history")
    ex = kits[
        (kits["component_type"] == "compressor")
        & (kits["size_class"] == "L")
        & (kits["refrigerant"] == "R32")
    ][["failure_mode", "part_no", "n_used", "n_repairs", "p_used", "qty_mean",
       "expected_qty_per_repair"]]
    print(ex.round(3).to_string(index=False))

    if args.validate:
        cmp_df = validate(kits, pd.read_parquet(args.data / "bom.parquet"))
        if cmp_df.empty:
            print("\nno overlap with ground truth to validate against")
        else:
            print(
                f"\nrecovery vs generator truth over {len(cmp_df):,} (mode, part) pairs:"
                f"\n  mean |error| in P(used) : {cmp_df['abs_err'].mean():.3f}"
                f"\n  median |error|          : {cmp_df['abs_err'].median():.3f}"
                f"\n  correlation             : {cmp_df['p_used'].corr(cmp_df['p_used_truth']):.3f}"
            )
            worst = cmp_df.nlargest(5, "abs_err")[
                ["component_type", "failure_mode", "part_no", "p_used", "p_used_truth", "n_repairs"]
            ]
            print("\n  worst-recovered pairs (usually the rarest modes):")
            print(worst.round(3).to_string(index=False))

    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
