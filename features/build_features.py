"""Feature engineering and the AFT snapshot sampler.

Two outputs:

  1. `asset_daily` - causal rolling features per asset-day. Every window looks
     strictly backwards, so a feature row dated T contains nothing from after T.
  2. `train` / `score` - the AFT matrices. One row per (component, snapshot),
     labelled with *remaining* time to the next failure and an interval-censoring
     pair. Labelling remaining rather than total life is what lets the fitted
     model be read directly as TTM, with no S(t+h)/S(t) renormalising.

The censoring is the part that matters. Most components in a fleet have not
failed yet; a plain regression that drops them learns a biased, too-short life,
which is precisely the failure mode A01 is trying to avoid. Censored rows are
kept with y_upper = +inf so XGBoost's AFT objective can use the information
"survived at least this long" without pretending to know when it will fail.

Run:
    python -m features.build_features
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"

# Signals we build rolling statistics over. Refrigerant-side signals are NaN
# for air-only machines (AHU); XGBoost handles missing natively, which is why
# we leave them as NaN rather than imputing a fake zero.
SIGNALS = [
    "suction_p", "discharge_p", "superheat", "subcool", "supply_t", "return_t",
    "comp_current", "vib_rms", "vib_kurtosis", "current_imbalance", "cop",
    "charge_frac", "ambient_t_max", "runtime_h", "load_frac", "starts", "short_cycles",
]

WINDOWS = (7, 30, 90)
SLOPE_WINDOWS = (30, 90)

# Train/test split: by time AND by asset, so neither a future month nor a
# sibling component on the same machine can leak into training.
TRAIN_DAYS = 547  # months 1-18
HOLDOUT_ASSET_FRAC = 0.20
SNAPSHOT_EVERY_DAYS = 30
WARMUP_DAYS = 95  # need a full 90-day window before the first snapshot


# --------------------------------------------------------------------------
# Rolling primitives
# --------------------------------------------------------------------------


def _slope_weights(w: int) -> np.ndarray:
    """Weights that turn a rolling dot-product into an exact OLS slope.

    For evenly spaced x, slope = sum((x - xbar) * y) / sum((x - xbar)^2). The
    denominator is constant for a fixed window, so the whole rolling slope is
    one convolution - far cheaper than rolling().apply(polyfit) and exactly
    equal to it.
    """
    x = np.arange(w, dtype=float)
    xc = x - x.mean()
    return xc / np.sum(xc**2)


def rolling_stats(v: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling mean and std, NaN-tolerant, same length as input."""
    s = pd.Series(v)
    r = s.rolling(w, min_periods=max(2, w // 4))
    return r.mean().to_numpy(), r.std().to_numpy()


def rolling_slope(v: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling OLS slope (units per day), NaN-tolerant.

    Implemented as one convolution rather than rolling().apply(): the slope is
    a fixed-weight dot product over the window, so convolving with the reversed
    weight vector gives exactly the same numbers in O(n) instead of one Python
    call per window position. At 500 assets x 17 signals x 2 windows x 730 days
    that is the difference between seconds and many minutes.
    """
    s = pd.Series(v)
    # Interpolate only *within* the series for the slope fit so a short gateway
    # dropout does not blank a 90-day trend; the raw NaNs are preserved
    # elsewhere and surfaced as a data-quality feature.
    filled = s.interpolate(limit_direction="both", limit=10).to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(filled) < w or not np.isfinite(filled).any():
        return out
    filled = np.nan_to_num(filled, nan=float(np.nanmean(filled)))
    wts = _slope_weights(w)
    # np.convolve reverses the kernel, so pass it reversed to get a plain
    # window dot-product aligned to the window's right edge.
    out[w - 1 :] = np.convolve(filled, wts[::-1], mode="valid")
    return out


# --------------------------------------------------------------------------
# Asset-level daily features
# --------------------------------------------------------------------------


def build_asset_features(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Causal rolling features per asset-day."""
    telemetry = telemetry.sort_values(["asset_id", "ts"])
    frames: list[pd.DataFrame] = []

    for asset_id, g in telemetry.groupby("asset_id", sort=False):
        n = len(g)
        # Accumulate into a dict and build the frame once. Assigning ~125
        # columns one at a time fragments the block manager and triggers a
        # PerformanceWarning per insert (70s and 4.8MB of warnings at fleet
        # scale); one concat at the end is both quiet and faster.
        cols: dict[str, np.ndarray] = {}
        nan_col = np.full(n, np.nan)

        for sig in SIGNALS:
            if sig not in g.columns:
                continue
            v = g[sig].to_numpy(dtype=float)
            if np.all(np.isnan(v)):
                # Air-only machine: leave the columns present-but-NaN rather
                # than fabricating zeros the model would read as real readings.
                for w in WINDOWS:
                    cols[f"{sig}_mean_{w}d"] = nan_col
                    if w >= 30:
                        cols[f"{sig}_std_{w}d"] = nan_col
                for w in SLOPE_WINDOWS:
                    cols[f"{sig}_slope_{w}d"] = nan_col
                continue
            for w in WINDOWS:
                m, sd = rolling_stats(v, w)
                cols[f"{sig}_mean_{w}d"] = m
                if w >= 30:
                    cols[f"{sig}_std_{w}d"] = sd
            for w in SLOPE_WINDOWS:
                cols[f"{sig}_slope_{w}d"] = rolling_slope(v, w)

        # Data quality, carried as features so the confidence badge has inputs
        core = [s for s in ("discharge_p", "comp_current", "vib_rms", "supply_t") if s in g.columns]
        if core:
            miss = g[core].isna().to_numpy(dtype=float)
            cols["imputed_share_30d"] = (
                pd.Series(miss.mean(axis=1)).rolling(30, min_periods=1).mean().to_numpy()
            )
            seen = np.where(g[core[0]].isna().to_numpy(), np.nan, np.arange(n))
            ffilled = pd.Series(seen).ffill().to_numpy()
            cols["hours_since_reading"] = (np.arange(n) - np.nan_to_num(ffilled, nan=0.0)) * 24.0
        else:
            cols["imputed_share_30d"] = np.zeros(n)
            cols["hours_since_reading"] = np.zeros(n)

        # Derived thermodynamics: head pressure normalised by ambient, and the
        # air-side delta-T, are the physically meaningful fouling indicators -
        # either raw signal alone is confounded by how hot the day was.
        if "discharge_p" in g.columns and "ambient_t_max" in g.columns:
            dp = g["discharge_p"].to_numpy(dtype=float)
            amb = g["ambient_t_max"].to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                cols["head_pressure_per_ambient"] = dp / np.where(amb > 0, amb, np.nan)
        if "return_t" in g.columns and "supply_t" in g.columns:
            cols["delta_t_air"] = (
                g["return_t"].to_numpy(dtype=float) - g["supply_t"].to_numpy(dtype=float)
            )
        if "ambient_t_max" in g.columns:
            # Degree-days above the design point: the standard HVAC stress proxy
            dd = np.clip(g["ambient_t_max"].to_numpy(dtype=float) - 35.0, 0, None)
            cols["degree_days_30d"] = pd.Series(dd).rolling(30, min_periods=1).sum().to_numpy()
            cols["degree_days_90d"] = pd.Series(dd).rolling(90, min_periods=1).sum().to_numpy()
        if "starts" in g.columns:
            cols["starts_mean_7d"] = (
                pd.Series(g["starts"].to_numpy(dtype=float)).rolling(7, min_periods=1).mean().to_numpy()
            )

        out = pd.DataFrame(cols)
        out.insert(0, "ts", g["ts"].to_numpy())
        out.insert(0, "asset_id", asset_id)
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Snapshot sampler / AFT labelling
# --------------------------------------------------------------------------


def build_snapshots(
    components: pd.DataFrame,
    work_orders: pd.DataFrame,
    asset_feat: pd.DataFrame,
    assets: pd.DataFrame,
    as_of: pd.Timestamp,
    window_start: pd.Timestamp,
) -> pd.DataFrame:
    """One row per (component, snapshot date) with interval-censored labels.

    Label convention for XGBoost's survival:aft objective:
        uncensored  y_lower = y_upper = days until the observed failure
        censored    y_lower = days observed so far, y_upper = +inf
    """
    cm = work_orders[(work_orders["type"] != "PM") & work_orders["component_id"].notna()]
    fail_by_comp: dict[str, np.ndarray] = {
        cid: np.sort(grp["open_ts"].to_numpy()) for cid, grp in cm.groupby("component_id")
    }
    pm = work_orders[work_orders["type"] == "PM"]
    pm_by_asset: dict[str, np.ndarray] = {
        aid: np.sort(grp["open_ts"].to_numpy()) for aid, grp in pm.groupby("asset_id")
    }

    snap_dates = pd.date_range(
        window_start + pd.Timedelta(days=WARMUP_DAYS), as_of, freq=f"{SNAPSHOT_EVERY_DAYS}D"
    )
    asset_meta = assets.set_index("asset_id")

    rows: list[dict] = []
    for comp in components.itertuples(index=False):
        fails = fail_by_comp.get(comp.component_id, np.array([], dtype="datetime64[ns]"))
        pms = pm_by_asset.get(comp.asset_id, np.array([], dtype="datetime64[ns]"))
        meta = asset_meta.loc[comp.asset_id]

        for snap in snap_dates:
            snap64 = np.datetime64(snap)
            nxt = fails[fails > snap64]
            if len(nxt):
                dur = float((nxt[0] - snap64) / np.timedelta64(1, "D"))
                y_lower = y_upper = max(dur, 0.5)
                event = True
            else:
                dur = float((np.datetime64(as_of) - snap64) / np.timedelta64(1, "D"))
                if dur < 1.0:
                    continue  # nothing observed after this snapshot
                y_lower, y_upper = dur, np.inf
                event = False

            # Component state as of the snapshot - strictly backward looking
            prev_fails = fails[fails <= snap64]
            last_event = prev_fails[-1] if len(prev_fails) else np.datetime64(pd.Timestamp(comp.install_date))
            prev_pms = pms[pms <= snap64]
            last_pm = prev_pms[-1] if len(prev_pms) else last_event

            row: dict = {
                "component_id": comp.component_id,
                "asset_id": comp.asset_id,
                "component_type": comp.component_type,
                "snapshot": snap,
                "y_lower": y_lower,
                "y_upper": y_upper,
                "event": event,
                "age_days": float((snap64 - last_event) / np.timedelta64(1, "D")),
                "days_since_pm": float((snap64 - last_pm) / np.timedelta64(1, "D")),
                "n_past_repairs": int(len(prev_fails)),
                "asset_age_days": float(meta["age_days"]),
                "capacity_kw": float(meta["capacity_kw"]),
                "nominal_duty_h": float(meta["nominal_duty_h"]),
                "asset_type": meta["asset_type"],
                "size_class": meta["size_class"],
                "climate_zone": meta["climate_zone"],
                "criticality": meta["criticality"],
                "hub_id": meta["hub_id"],
                "region": meta["region"],
            }
            rows.append(row)

    snaps = pd.DataFrame(rows)
    if snaps.empty:
        return snaps
    # Attach the rolling features with one merge rather than a per-row .loc on
    # a MultiIndex - same result, and it turns ~120k index lookups into a
    # single hash join.
    return snaps.merge(
        asset_feat, left_on=["asset_id", "snapshot"], right_on=["asset_id", "ts"], how="inner"
    ).drop(columns=["ts"])


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot the categoricals and return (frame, feature column list)."""
    cats = ["component_type", "asset_type", "size_class", "climate_zone", "criticality"]
    out = df.copy()
    for c in cats:
        if c in out.columns:
            out = pd.concat([out, pd.get_dummies(out[c], prefix=c, dtype=float)], axis=1)
    drop = {
        "component_id", "asset_id", "snapshot", "y_lower", "y_upper", "event",
        "hub_id", "region", *cats,
    }
    feats = [c for c in out.columns if c not in drop and out[c].dtype.kind in "fiub"]
    return out, feats


def split(df: pd.DataFrame, window_start: pd.Timestamp, seed: int = 42) -> pd.DataFrame:
    """Tag rows train/test by time and by asset.

    A row is test if it is in the final 6 months OR belongs to a held-out
    asset. Both, because a sibling component on the same machine shares its
    telemetry history, so an asset-blind time split would still leak.
    """
    rng = np.random.default_rng(seed)
    assets = np.sort(df["asset_id"].unique())
    n_hold = int(len(assets) * HOLDOUT_ASSET_FRAC)
    held = set(rng.choice(assets, size=n_hold, replace=False).tolist())

    day = (df["snapshot"] - window_start).dt.days
    df = df.copy()
    df["held_out_asset"] = df["asset_id"].isin(held)
    df["late_period"] = day > TRAIN_DAYS
    df["fold"] = np.where(df["held_out_asset"] | df["late_period"], "test", "train")
    return df


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Build features and AFT training matrices")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.perf_counter()
    d = args.data
    telemetry = pd.read_parquet(d / "telemetry_daily.parquet")
    components = pd.read_parquet(d / "components.parquet")
    work_orders = pd.read_parquet(d / "work_orders.parquet")
    assets = pd.read_parquet(d / "assets.parquet")

    as_of = pd.Timestamp(telemetry["ts"].max())
    window_start = pd.Timestamp(telemetry["ts"].min())
    print(f"window {window_start.date()} .. {as_of.date()}  ({len(telemetry):,} telemetry rows)")

    print("building asset rolling features ...")
    asset_feat = build_asset_features(telemetry)
    asset_feat.to_parquet(d / "asset_features.parquet", index=False)
    print(f"  {len(asset_feat):,} asset-days x {asset_feat.shape[1] - 2} features"
          f"  ({time.perf_counter() - t0:.1f}s)")

    print("sampling snapshots and labelling ...")
    snaps = build_snapshots(components, work_orders, asset_feat, assets, as_of, window_start)
    snaps = split(snaps, window_start, seed=args.seed)
    enc, feats = encode(snaps)
    enc.to_parquet(d / "train_snapshots.parquet", index=False)
    pd.Series(feats).to_frame("feature").to_parquet(d / "feature_list.parquet", index=False)

    tr = enc[enc["fold"] == "train"]
    te = enc[enc["fold"] == "test"]
    print(
        f"  {len(enc):,} rows, {len(feats)} features\n"
        f"    train {len(tr):,}  ({int(tr['event'].sum()):,} events, "
        f"{1 - tr['event'].mean():.0%} censored)\n"
        f"    test  {len(te):,}  ({int(te['event'].sum()):,} events, "
        f"{1 - te['event'].mean():.0%} censored)"
    )

    # The scoring matrix: the latest snapshot per component, which is what the
    # pipeline predicts on to produce "today's" TTM for every component.
    latest = enc.sort_values("snapshot").groupby("component_id", as_index=False).tail(1)
    latest.to_parquet(d / "score_snapshots.parquet", index=False)
    print(f"  score matrix: {len(latest):,} components at {as_of.date()}")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
