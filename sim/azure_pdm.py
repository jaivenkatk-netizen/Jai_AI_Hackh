"""Microsoft/Azure Predictive Maintenance dataset loader.

Why this dataset specifically: of the public sets available, it is the only one
whose *shape* matches A01 - component-level failures plus a maintenance history
on the same machines. 100 machines x 4 components, hourly telemetry (volt,
rotate, pressure, vibration), error logs, replacement records and 761 labelled
failures. That lets the exact same survival pipeline be scored on real failure
history, which is the honest answer to "your synthetic data matches its own
model".

It is not HVAC, and the pitch says so. Its job is to validate the *method*
(censoring handled correctly, AFT calibration holds on real data), while the
synthetic fleet supplies the HVAC physics and the parts/stock/supplier links
that no public set provides.

Output is deliberately shaped like the synthetic tables - daily telemetry
aggregates plus a component-life table with an event/censoring flag - so
models/survival_xgb.py can train on either source with no special-casing.

The canonical Azure blob host (azuremlsampleexperiments.blob.core.windows.net)
has been retired; DNS no longer resolves. We read a GitHub mirror instead and
degrade gracefully when offline, because a demo must never depend on a network
call succeeding.

Run:
    python -m sim.azure_pdm              # download (cached), transform, write Parquet
    python -m sim.azure_pdm --check      # report availability only
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

MIRROR = "https://raw.githubusercontent.com/AbdelhafidhMahouel/predictive-maintenance-db/main/data"
FILES = ("PdM_machines.csv", "PdM_failures.csv", "PdM_maint.csv", "PdM_errors.csv", "PdM_telemetry.csv")

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "azure_pdm"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"

COMPONENTS = ("comp1", "comp2", "comp3", "comp4")
SENSORS = ("volt", "rotate", "pressure", "vibration")


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def _download(name: str, dest: Path, timeout: float = 120.0) -> bool:
    url = f"{MIRROR}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "servicesense/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            last = 0.0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total and time.monotonic() - last > 1.0:
                    print(f"\r    {name}: {got / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
                    last = time.monotonic()
        tmp.replace(dest)
        print(f"\r    {name}: {got / 1e6:6.1f} MB  done" + " " * 12)
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"\r    {name}: FAILED ({exc})" + " " * 12)
        return False


def fetch(force: bool = False) -> bool:
    """Download any missing files into the raw cache. True if all present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in FILES:
        dest = RAW_DIR / name
        if dest.exists() and not force:
            print(f"    {name}: cached ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        ok &= _download(name, dest)
    return ok


def available() -> bool:
    return all((RAW_DIR / n).exists() for n in FILES)


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------


def build_component_lives(maint: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct each component's lives, flagging event vs censoring.

    The dataset records a replacement in PdM_maint every time a component is
    changed, and separately records in PdH_failures whether that change was a
    *failure* or a proactive swap. So a life runs from one replacement to the
    next, and ends in an event only if a failure is logged at the closing date.
    A component's final life is right-censored at the end of the observation
    window - which is the whole reason this has to be a survival model and not
    a regression: dropping those rows would bias every predicted life short.
    """
    maint = maint.rename(columns={"comp": "component"}).copy()
    maint["datetime"] = pd.to_datetime(maint["datetime"])
    failures = failures.rename(columns={"failure": "component"}).copy()
    failures["datetime"] = pd.to_datetime(failures["datetime"])

    fail_keys = set(zip(failures["machineID"], failures["component"], failures["datetime"]))
    obs_end = max(maint["datetime"].max(), failures["datetime"].max())

    # Every replacement is a life boundary; union with failures in case a
    # failure was logged without a matching maint row.
    bounds = pd.concat(
        [maint[["machineID", "component", "datetime"]], failures[["machineID", "component", "datetime"]]],
        ignore_index=True,
    ).drop_duplicates()

    rows = []
    for (machine, comp), grp in bounds.groupby(["machineID", "component"], sort=True):
        dts = sorted(grp["datetime"].unique())
        # Life 0 starts at the window open; we do not know the true install
        # date, so it is left-truncated and excluded from training below.
        prev = None
        for dt in dts:
            if prev is not None:
                rows.append(
                    {
                        "machineID": machine,
                        "component": comp,
                        "start": prev,
                        "end": dt,
                        "duration_days": (dt - prev).total_seconds() / 86400.0,
                        "event": (machine, comp, dt) in fail_keys,
                        "left_truncated": False,
                    }
                )
            prev = dt
        if prev is not None and prev < obs_end:
            rows.append(
                {
                    "machineID": machine,
                    "component": comp,
                    "start": prev,
                    "end": obs_end,
                    "duration_days": (obs_end - prev).total_seconds() / 86400.0,
                    "event": False,  # still running: right-censored
                    "left_truncated": False,
                }
            )
    lives = pd.DataFrame(rows)
    lives = lives[lives["duration_days"] > 0.5].reset_index(drop=True)
    return lives


def daily_telemetry(telemetry: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly telemetry to daily, matching the synthetic schema.

    Mean/std/min/max per sensor per day, plus daily error counts. Daily is the
    right grain because the feature pipeline computes rolling windows per
    component-day; keeping 876k hourly rows would cost memory for no gain.
    """
    telemetry = telemetry.copy()
    telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
    telemetry["date"] = telemetry["datetime"].dt.floor("D")

    agg = {s: ["mean", "std", "min", "max"] for s in SENSORS}
    daily = telemetry.groupby(["machineID", "date"]).agg(agg)
    daily.columns = [f"{s}_{stat}" for s, stat in daily.columns]
    daily = daily.reset_index()

    errors = errors.copy()
    errors["datetime"] = pd.to_datetime(errors["datetime"])
    errors["date"] = errors["datetime"].dt.floor("D")
    err_counts = (
        errors.assign(n=1)
        .pivot_table(index=["machineID", "date"], columns="errorID", values="n", aggfunc="sum")
        .fillna(0)
        .reset_index()
    )
    err_counts.columns = [
        c if c in ("machineID", "date") else f"n_{c}" for c in err_counts.columns
    ]

    out = daily.merge(err_counts, on=["machineID", "date"], how="left")
    err_cols = [c for c in out.columns if c.startswith("n_error")]
    out[err_cols] = out[err_cols].fillna(0)
    out["n_errors_total"] = out[err_cols].sum(axis=1) if err_cols else 0.0
    return out


def build_training_table(
    lives: pd.DataFrame, daily: pd.DataFrame, machines: pd.DataFrame
) -> pd.DataFrame:
    """Snapshot sampler: one row per (component life, snapshot) for AFT training.

    Mirrors what features/build_features.py does on the synthetic fleet - take
    a snapshot inside a component's life, compute features from the trailing
    window only, and label the *remaining* time to the event. Labelling
    remaining rather than total life is what lets the AFT model be read
    directly as TTM without renormalising by current age.
    """
    daily = daily.sort_values(["machineID", "date"])
    machines = machines.copy()

    rows = []
    # Snapshot every 7 days inside each life, keeping a 15-day warm-up so the
    # rolling features have history to work with.
    for life in lives.itertuples(index=False):
        md = daily[daily["machineID"] == life.machineID]
        if md.empty:
            continue
        start, end = life.start, life.end
        snap = start + pd.Timedelta(days=15)
        while snap < end:
            win = md[(md["date"] > snap - pd.Timedelta(days=30)) & (md["date"] <= snap)]
            if len(win) >= 7:
                feat: dict[str, float | str | bool] = {
                    "machineID": life.machineID,
                    "component": life.component,
                    "snapshot": snap,
                    "age_days": (snap - start).total_seconds() / 86400.0,
                    "remaining_days": (end - snap).total_seconds() / 86400.0,
                    "event": bool(life.event),
                }
                for s in SENSORS:
                    col = f"{s}_mean"
                    if col in win:
                        v = win[col].to_numpy(dtype=float)
                        feat[f"{s}_mean_30d"] = float(np.nanmean(v))
                        feat[f"{s}_std_30d"] = float(np.nanstd(v))
                        # Trend is the signal that matters for degradation
                        if len(v) >= 3 and np.isfinite(v).sum() >= 3:
                            idx = np.arange(len(v), dtype=float)
                            ok = np.isfinite(v)
                            feat[f"{s}_slope_30d"] = float(np.polyfit(idx[ok], v[ok], 1)[0])
                        else:
                            feat[f"{s}_slope_30d"] = 0.0
                    sd = f"{s}_std"
                    if sd in win:
                        feat[f"{s}_vol_30d"] = float(np.nanmean(win[sd].to_numpy(dtype=float)))
                feat["n_errors_30d"] = float(win.get("n_errors_total", pd.Series([0.0])).sum())
                rows.append(feat)
            snap = snap + pd.Timedelta(days=7)

    train = pd.DataFrame(rows)
    if train.empty:
        return train
    train = train.merge(machines, on="machineID", how="left")
    # One-hot the machine model; keep age as given (years, in this dataset)
    if "model" in train:
        train = pd.concat([train, pd.get_dummies(train["model"], prefix="model")], axis=1)
    train = pd.concat([train, pd.get_dummies(train["component"], prefix="comp")], axis=1)
    return train


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def load_and_transform(force: bool = False) -> dict[str, pd.DataFrame] | None:
    if not fetch(force=force):
        print(
            "\n  Azure PdM download incomplete. The pipeline continues with the\n"
            "  synthetic fleet only; public-data validation will be skipped.\n"
            f"  To supply it manually, drop the 5 CSVs into {RAW_DIR}\n"
            "  (Kaggle: arnabbiswas1/microsoft-azure-predictive-maintenance).",
            file=sys.stderr,
        )
        return None

    print("  reading CSVs ...")
    machines = pd.read_csv(RAW_DIR / "PdM_machines.csv")
    failures = pd.read_csv(RAW_DIR / "PdM_failures.csv")
    maint = pd.read_csv(RAW_DIR / "PdM_maint.csv")
    errors = pd.read_csv(RAW_DIR / "PdM_errors.csv")
    telemetry = pd.read_csv(RAW_DIR / "PdM_telemetry.csv")

    print(f"    machines {len(machines):,} | failures {len(failures):,} | "
          f"maint {len(maint):,} | errors {len(errors):,} | telemetry {len(telemetry):,}")

    print("  reconstructing component lives ...")
    lives = build_component_lives(maint, failures)
    print(f"    {len(lives):,} lives, {int(lives['event'].sum()):,} failures, "
          f"{int((~lives['event']).sum()):,} censored "
          f"({(~lives['event']).mean():.0%} censoring rate)")

    print("  aggregating telemetry to daily ...")
    daily = daily_telemetry(telemetry, errors)
    print(f"    {len(daily):,} machine-days")

    print("  building snapshot training table ...")
    train = build_training_table(lives, daily, machines)
    print(f"    {len(train):,} training rows, {int(train['event'].sum()):,} uncensored")

    return {
        "pdm_machines": machines,
        "pdm_lives": lives,
        "pdm_daily": daily,
        "pdm_train": train,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Load and transform the Azure PdM dataset")
    ap.add_argument("--check", action="store_true", help="report cache status and exit")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.check:
        print(f"raw cache: {RAW_DIR}")
        for n in FILES:
            p = RAW_DIR / n
            print(f"  {'OK  ' if p.exists() else '--  '}{n}"
                  + (f"  {p.stat().st_size / 1e6:.1f} MB" if p.exists() else ""))
        print(f"\ncomplete: {available()}")
        return

    t0 = time.perf_counter()
    tables = load_and_transform(force=args.force)
    if tables is None:
        sys.exit(0)  # not an error: synthetic-only is a supported mode

    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = args.out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {name:16s} {len(df):>9,} rows -> {path.name}")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
