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

# 30/90-day windows only. The 7-day window added 17 features that duplicate
# what the 30-day mean already carries for a degradation signal measured over
# months, and this box has 3.9GB of RAM - feature count is a memory budget here,
# not a free parameter.
WINDOWS = (30, 90)
SLOPE_WINDOWS = (30, 90)

# Extra snapshots taken this many days before each observed failure.
#
# A uniform 30-day grid almost never lands close to a failure, so the model
# sees very few short-horizon labels and learns to hedge: measured, the
# minimum predicted P50 across the whole fleet was 19 days, which leaves the
# Urgent bucket (0-14 d) permanently empty - the one bucket the brief cares
# most about. Anchoring a few snapshots just before each event teaches the
# short-horizon regime directly, at a fraction of the cost of densifying the
# whole grid.
EVENT_ANCHORED_OFFSETS = (3, 9, 18)

# Train/test split: by time AND by asset, so neither a future month nor a
# sibling component on the same machine can leak into training.
# Expressed as a fraction of the window rather than a fixed day count - a
# hardcoded 547 days was 75% of the original 2-year window but only 30% of a
# 5-year one, which silently pushed most of the data into test.
TRAIN_TIME_FRAC = 0.75
HOLDOUT_ASSET_FRAC = 0.20
SNAPSHOT_EVERY_DAYS = 45
WARMUP_DAYS = 95  # need a full 90-day window before the first snapshot


# --------------------------------------------------------------------------
# Rolling primitives
# --------------------------------------------------------------------------


def snapshot_dates(window_start: pd.Timestamp, as_of: pd.Timestamp) -> pd.DatetimeIndex:
    """The snapshot grid, defined in exactly one place.

    Anchored to end at `as_of` so today's prediction uses today's telemetry.
    Both the feature filter and the labeller must use the same grid - when they
    were computed independently the two ranges drifted apart and the inner
    merge silently produced zero rows.
    """
    span = (as_of - (window_start + pd.Timedelta(days=WARMUP_DAYS))).days
    n = int(span // SNAPSHOT_EVERY_DAYS) + 1
    return pd.date_range(end=as_of, periods=max(n, 2), freq=f"{SNAPSHOT_EVERY_DAYS}D")


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


def event_anchored_dates(
    work_orders: pd.DataFrame, window_start: pd.Timestamp, as_of: pd.Timestamp
) -> dict[str, set[pd.Timestamp]]:
    """Per-asset snapshot dates sitting just before each observed failure."""
    cm = work_orders[(work_orders["type"] != "PM") & work_orders["component_id"].notna()]
    lo = window_start + pd.Timedelta(days=WARMUP_DAYS)
    out: dict[str, set[pd.Timestamp]] = {}
    for aid, grp in cm.groupby("asset_id"):
        dates: set[pd.Timestamp] = set()
        for ts in grp["open_ts"]:
            for off in EVENT_ANCHORED_OFFSETS:
                d = pd.Timestamp(ts).normalize() - pd.Timedelta(days=off)
                if lo <= d <= as_of:
                    dates.add(d)
        if dates:
            out[str(aid)] = dates
    return out


def build_asset_features(
    telemetry: pd.DataFrame,
    keep_dates: pd.DatetimeIndex | None = None,
    keep_by_asset: dict[str, set[pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Causal rolling features per asset-day.

    `keep_dates` restricts the *returned* rows (rolling windows still see the
    full history, so nothing is lost). Over a 5-year window the full matrix is
    912,500 x 125, which needs ~900MB per array and OOMs on concat - and every
    one of those rows except the ~60 snapshot dates per asset is discarded
    immediately afterwards. Filtering inside the loop cuts the result to ~30k
    rows at identical values.

    Features are emitted as float32: the downstream consumer is XGBoost, which
    casts to float32 internally anyway, so float64 doubles peak memory for no
    precision that survives.
    """
    telemetry = telemetry.sort_values(["asset_id", "ts"])
    frames: list[pd.DataFrame] = []
    keep = set(keep_dates) if keep_dates is not None else None

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

        out = pd.DataFrame({k: v.astype(np.float32, copy=False) for k, v in cols.items()})
        out.insert(0, "ts", g["ts"].to_numpy())
        out.insert(0, "asset_id", asset_id)
        if keep is not None:
            wanted = keep | (keep_by_asset.get(str(asset_id), set()) if keep_by_asset else set())
            out = out[out["ts"].isin(wanted)]
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

    snap_dates = snapshot_dates(window_start, as_of)
    asset_meta = assets.set_index("asset_id")

    rows: list[dict] = []
    lo = window_start + pd.Timedelta(days=WARMUP_DAYS)
    for comp in components.itertuples(index=False):
        fails = fail_by_comp.get(comp.component_id, np.array([], dtype="datetime64[ns]"))
        pms = pm_by_asset.get(comp.asset_id, np.array([], dtype="datetime64[ns]"))
        meta = asset_meta.loc[comp.asset_id]

        # The uniform grid, plus a few snapshots just before this component's
        # own failures so short-horizon labels are actually represented.
        comp_dates = set(snap_dates)
        for f in fails:
            for off in EVENT_ANCHORED_OFFSETS:
                d = pd.Timestamp(f).normalize() - pd.Timedelta(days=off)
                if lo <= d <= as_of:
                    comp_dates.add(d)

        for snap in sorted(comp_dates):
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

            # Component state as of the snapshot - strictly backward looking.
            #
            # `components.install_date` must NOT be used here: it holds the
            # component's *most recent* service date, which for a part replaced
            # later in the window lies in the snapshot's future. Using it made
            # age_days negative (median -16 d on near-failure rows, -936 d
            # overall) and leaked the replacement date into the features. Since
            # age_days is a top-gain feature, that corruption swamped the real
            # telemetry signal. The honest fallback is the asset's commissioning
            # date, which is genuinely known at snapshot time.
            prev_fails = fails[fails <= snap64]
            last_event = (
                prev_fails[-1] if len(prev_fails) else np.datetime64(pd.Timestamp(meta["install_date"]))
            )
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
                # Clipped at 0: an age can never be negative, and a stray
                # negative would be a silent signal that the leak is back.
                "age_days": max(0.0, float((snap64 - last_event) / np.timedelta64(1, "D"))),
                "days_since_pm": max(0.0, float((snap64 - last_pm) / np.timedelta64(1, "D"))),
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
    cutoff = float(day.max()) * TRAIN_TIME_FRAC
    df = df.copy()
    df["held_out_asset"] = df["asset_id"].isin(held)
    df["late_period"] = day > cutoff
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

    # Snapshot dates are known up front, so the feature matrix only needs to
    # materialise those rows (rolling windows still read the whole series).
    snap_dates = snapshot_dates(window_start, as_of)
    anchored = event_anchored_dates(work_orders, window_start, as_of)
    n_anchored = sum(len(v) for v in anchored.values())
    print(f"building asset rolling features at {len(snap_dates)} grid dates "
          f"({snap_dates[0].date()} .. {snap_dates[-1].date()}) "
          f"+ {n_anchored:,} event-anchored dates ...")
    asset_feat = build_asset_features(telemetry, keep_dates=snap_dates, keep_by_asset=anchored)
    asset_feat.to_parquet(d / "asset_features.parquet", index=False)
    print(f"  {len(asset_feat):,} asset-snapshots x {asset_feat.shape[1] - 2} features"
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
