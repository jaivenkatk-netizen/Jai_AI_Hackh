"""Ingest the three public real-world datasets dropped into data/raw.

They are NOT HVAC fleets with parts and hubs, so they do not feed the stock
model. Their job is the same as the Azure PdM set's: independent evidence that
the *method* holds on data we did not generate. Each answers one claim:

  ASHRAE RP-1043  (lab chiller, graded faults)
      Does a fault of known severity move the refrigerant-side signals the
      twin and the fault-state network rely on, and by how much? A baseline is
      fitted on the fault-free runs and the *residual* is reported, because the
      runs sit at different loads and a raw difference would be confounded.

  LBNL FDD chiller plant  (plant-level simulation, 1 minute, ~1 year each)
      What is the energy cost of each plant fault? Reported as % change in
      plant power per unit cooling load against the fault-free year.

  NASA IMS bearings  (run-to-failure vibration)
      The only real remaining-life curves here. Reports detection lead time
      before failure and the rank correlation between a health indicator and
      remaining life, per test, the same honesty check the survival model gets
      per component type.

Outputs are compact Parquet in data/synthetic (committed); the raw archives
and their extraction stay out of git (data/raw/ is ignored).

Memory: this box has ~3.9GB of RAM. LBNL CSVs are ~430MB each, so they are
streamed from the zip in 32MB blocks and reduced to daily means; IMS is read
one 20480x8 file at a time.

Run:
    python -m sim.real_data --source ashrae
    python -m sim.real_data --source lbnl
    python -m sim.real_data --source ims
    python -m sim.real_data              # all three
"""

from __future__ import annotations

import argparse
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXTRACTED = RAW / "extracted"
OUT = ROOT / "data" / "synthetic"

# ---------------------------------------------------------------- ASHRAE ----

ASHRAE_DIR = EXTRACTED / "ashrae" / "Chiller Data"
ASHRAE_COLS = [
    "TEO", "TCI", "Evap Tons", "kW", "COP", "kW/Ton", "PRE", "PRC",
    "TRC_sub", "Tsh_suc", "Tsh_dis", "TEA", "TCA", "Amps", "TO_sump",
]
ZSIGNALS = ["kW/Ton", "TRC_sub", "Tsh_suc", "Tsh_dis", "PRC", "PRE", "TCA", "TEA"]
ASHRAE_FAULTS = {
    "Condenser fouling": "Condenser fouling",
    "Reduced condenser water flow": "Reduced condenser flow",
    "Reduced evaporator water flow": "Reduced evaporator flow",
    "Refrigerant leak": "Refrigerant leak",
    "Refrigerant overcharge": "Refrigerant overcharge",
    "Excess oil": "Excess oil",
    "Non-condensables in refrigerant": "Non-condensables",
}


def _severity(stem: str) -> float | None:
    m = re.search(r"(\d+)", stem)
    return float(m.group(1)) if m else None


def build_ashrae() -> pd.DataFrame:
    import xlrd

    if not ASHRAE_DIR.exists():
        raise SystemExit(f"{ASHRAE_DIR} missing - extract 'ASHRAE RP-1043.zip' first")

    rows = []

    def read(path: Path, fault: str, sev: float | None):
        try:
            book = xlrd.open_workbook(str(path), on_demand=True)
            sh = book.sheet_by_name("Steady States")
        except Exception as e:  # unreadable / aborted-run files exist in the set
            print(f"  skip {path.name}: {e}")
            return
        head = [str(c).strip() for c in sh.row_values(0)]
        idx = {c: head.index(c) for c in ASHRAE_COLS if c in head}
        for r in range(1, sh.nrows):
            v = sh.row_values(r)
            rec = {c: v[i] for c, i in idx.items()}
            rec["test"] = v[0]
            rec.update(fault=fault, severity=sev, file=path.stem)
            rows.append(rec)
        book.release_resources()

    for p in sorted((ASHRAE_DIR / "Benchmark Tests").glob("*.xls")):
        if p.stem.lower().startswith(("normal", "near normal")):
            read(p, "Normal", 0.0)
    for folder, label in ASHRAE_FAULTS.items():
        for p in sorted((ASHRAE_DIR / folder).glob("*.xls")):
            if "aborted" in p.stem or "trace" in p.stem:
                continue
            read(p, label, _severity(p.stem))

    df = pd.DataFrame(rows)
    for c in ASHRAE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["kW/Ton", "Evap Tons", "TEO", "TCI"])

    # Fault-free baseline per signal: a quadratic in the operating point
    # (load, evaporator-out and condenser-in water temperature), fitted on the
    # Normal runs only. The residual, in baseline standard deviations (z), is
    # what a fault did - the runs sit at different loads, so raw values are
    # confounded.
    def design(d: pd.DataFrame) -> np.ndarray:
        t, e, c = d["Evap Tons"], d["TEO"], d["TCI"]
        return np.c_[np.ones(len(d)), t, e, c, t * t, t * e, t * c]

    norm = df.fault == "Normal"
    for sig in ZSIGNALS:
        ok = df[sig].notna()
        fit = norm & ok
        beta, *_ = np.linalg.lstsq(design(df[fit]), df.loc[fit, sig].to_numpy(), rcond=None)
        pred = np.full(len(df), np.nan)
        pred[ok.to_numpy()] = design(df[ok]) @ beta
        res = df[sig].to_numpy() - pred
        sd = float(np.nanstd(res[fit.to_numpy()]))
        df[f"{sig}_z"] = res / sd
        if sig == "kW/Ton":
            df["kwton_expected"] = pred
            df["kwton_residual_pct"] = 100 * res / pred
    df["baseline_resid_sd_pct"] = float(df.loc[norm, "kwton_residual_pct"].std())
    print(f"  ashrae: {len(df)} steady states, {df.file.nunique()} runs, "
          f"baseline kW/Ton residual sd {df.baseline_resid_sd_pct.iloc[0]:.2f}%")
    return df


# ------------------------------------------------------------------ LBNL ----

LBNL_ZIP = RAW / "LBNL_FDD_Data_Sets_Chiller_Plant_all_3.zip"
POW_COLS = (
    [f"CHL_POW_{i}" for i in (1, 2, 3)]
    + [f"CT_POW_{i}" for i in (1, 2, 3)]
    + [f"CDWL_PM_POW_{i}" for i in (1, 2, 3)]
    + [f"CWL_PRI_PM_POW_{i}" for i in (1, 2, 3)]
    + [f"CWL_SEC_PM_POW_{i}" for i in (1, 2)]
)
LBNL_COLS = (
    ["Datetime", "CWL_SEC_LOAD", "CWL_SEC_DP", "CWL_SEC_DPSPT", "OA_TEMP_WB"]
    + POW_COLS
    + [f"CT_SW_TEMP_{i}" for i in (1, 2, 3)]
    + [f"CT_STA_{i}" for i in (1, 2, 3)]
)


def _lbnl_label(stem: str) -> tuple[str, str, float | None]:
    """ChillerPlant_<component>_<fault>_<level>  ->  (component, fault, level)."""
    s = stem.replace("ChillerPlant_", "").replace("ChillerPlant", "")
    if not s:
        return "none", "Fault-free", 0.0
    parts = s.split("_")
    comp = parts[0]
    fault = parts[1] if len(parts) > 1 else "fault"
    m = re.search(r"-?\d+$", s)
    lvl = None
    if m:
        raw = m.group(0)
        lvl = float(raw) / (100 if len(raw.lstrip("-")) == 3 else 1)
    return comp, fault, lvl


def build_lbnl() -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.csv as pacsv

    if not LBNL_ZIP.exists():
        raise SystemExit(f"{LBNL_ZIP} missing")
    z = zipfile.ZipFile(LBNL_ZIP)
    names = sorted(n for n in z.namelist()
                   if n.endswith(".csv") and not n.startswith("__MACOSX"))
    frames = []
    for n in names:
        t0 = time.time()
        comp, fault, lvl = _lbnl_label(Path(n).stem)
        with z.open(n) as f:
            reader = pacsv.open_csv(
                f,
                read_options=pacsv.ReadOptions(block_size=32 << 20),
                convert_options=pacsv.ConvertOptions(
                    include_columns=LBNL_COLS,
                    column_types={"Datetime": pa.string()},
                ),
            )
            parts = []
            for batch in reader:
                d = batch.to_pandas()
                run = d[[f"CT_STA_{i}" for i in (1, 2, 3)]].to_numpy() > 0
                ct_sw = d[[f"CT_SW_TEMP_{i}" for i in (1, 2, 3)]].where(run).mean(axis=1)
                g = pd.DataFrame({
                    "day": d["Datetime"].str.slice(0, 10),
                    "plant_kw": d[POW_COLS].sum(axis=1),
                    "chiller_kw": d[[f"CHL_POW_{i}" for i in (1, 2, 3)]].sum(axis=1),
                    "load": d["CWL_SEC_LOAD"],
                    "dp_err": d["CWL_SEC_DP"] - d["CWL_SEC_DPSPT"],
                    "ct_approach": ct_sw - d["OA_TEMP_WB"],
                })
                parts.append(g.groupby("day").agg(["sum", "count"]))
        agg = pd.concat(parts).groupby(level=0).sum()
        out = pd.DataFrame({"day": agg.index})
        for c in ("plant_kw", "chiller_kw", "load", "dp_err", "ct_approach"):
            out[c] = (agg[(c, "sum")] / agg[(c, "count")]).to_numpy()
        out["scenario"] = Path(n).stem.replace("ChillerPlant_", "")
        if out.scenario.iloc[0] == "ChillerPlant":
            out["scenario"] = "fault_free"
        out["component"], out["fault"], out["level"] = comp, fault, lvl
        frames.append(out)
        print(f"  lbnl {out.scenario.iloc[0]:<42} {len(out)} days  {time.time() - t0:5.0f}s")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------- IMS ----

IMS_DIR = EXTRACTED / "ims" / "IMS"
# test -> (folder, sub, expected file count, failing channel indices, mode)
IMS_TESTS = {
    1: ("1st", "1st_test", 2156, [4, 6], "Bearing 3 inner race / bearing 4 roller"),
    2: ("2nd", "2nd_test", 984, [0], "Bearing 1 outer race"),
    3: ("3rd", "4th_test/txt", 6324, [2], "Bearing 3 outer race"),
}


def _ims_dir(folder: str, sub: str) -> Path | None:
    for cand in (IMS_DIR / folder / sub, IMS_DIR / folder):
        if cand.exists():
            fs = [p for p in cand.iterdir() if p.is_file()]
            if fs:
                return cand
    return None


def build_ims(only: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import kurtosis, spearmanr

    series, summary = [], []
    for test, (folder, sub, expected, fail_ch, mode) in IMS_TESTS.items():
        if only and test not in only:
            continue
        d = _ims_dir(folder, sub)
        if d is None:
            print(f"  ims test {test}: not extracted, skipped")
            continue
        files = sorted(p for p in d.iterdir() if p.is_file())
        if len(files) < expected * 0.98:
            print(f"  ims test {test}: only {len(files)}/{expected} files, skipped")
            continue
        t0 = time.time()
        recs = []
        for p in files:
            a = pd.read_csv(p, sep="\t", header=None, dtype=np.float32).to_numpy()
            rec = {"ts": pd.to_datetime(p.name, format="%Y.%m.%d.%H.%M.%S")}
            for c in range(a.shape[1]):
                x = a[:, c]
                rec[f"rms{c}"] = float(np.sqrt(np.mean(x * x)))
                rec[f"kurt{c}"] = float(kurtosis(x, fisher=False))
            recs.append(rec)
        df = pd.DataFrame(recs).sort_values("ts").reset_index(drop=True)
        df["hours"] = (df.ts - df.ts.iloc[0]).dt.total_seconds() / 3600
        life_h = float(df.hours.iloc[-1])
        df["rul_h"] = life_h - df.hours
        # Health indicator = RMS of the bearing(s) that ultimately failed.
        fc = [c for c in fail_ch if f"rms{c}" in df.columns]
        df["hi"] = df[[f"rms{c}" for c in fc]].mean(axis=1)
        df["hi_kurt"] = df[[f"kurt{c}" for c in fc]].mean(axis=1)

        # Alarm: rolling-median HI above baseline + 5 sigma for 3 consecutive
        # samples; baseline = first 10% of the run (healthy by construction).
        nb = max(20, len(df) // 10)
        base = df.hi.iloc[:nb]
        thr = float(base.mean() + 5 * base.std())
        smooth = df.hi.rolling(5, min_periods=1).median()
        over = (smooth > thr).to_numpy()
        alarm_i = next((i for i in range(nb, len(df) - 2) if over[i:i + 3].all()), None)
        lead_h = float(df.rul_h.iloc[alarm_i]) if alarm_i is not None else float("nan")
        rho = float(spearmanr(df.hi, df.rul_h).statistic)
        late = df.iloc[len(df) // 2:]
        rho_late = float(spearmanr(late.hi, late.rul_h).statistic)
        summary.append(dict(
            test=test, failure_mode=mode, files=len(df), life_hours=life_h,
            life_days=life_h / 24, baseline_hi=float(base.mean()),
            alarm_threshold=thr,
            alarm_hours=float(df.hours.iloc[alarm_i]) if alarm_i is not None else np.nan,
            lead_time_hours=lead_h, lead_time_days=lead_h / 24,
            lead_pct_of_life=100 * lead_h / life_h if life_h else np.nan,
            rank_corr_hi_vs_rul=rho, rank_corr_last_half=rho_late,
        ))
        step = max(1, len(df) // 400)  # keep each curve to ~400 points
        s = df.iloc[::step][["hours", "rul_h", "hi", "hi_kurt"]].copy()
        s["test"] = test
        s["alarm_threshold"] = thr
        series.append(s)
        print(f"  ims test {test}: {len(df)} files, life {life_h / 24:.1f} d, "
              f"alarm lead {lead_h / 24:.1f} d, rho {rho:+.2f}  {time.time() - t0:.0f}s")
    return (pd.concat(series, ignore_index=True) if series else pd.DataFrame(),
            pd.DataFrame(summary))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", type=int, nargs="*", help="IMS tests to (re)build and merge in")
    ap.add_argument("--source", choices=["ashrae", "lbnl", "ims", "all"], default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.source in ("ashrae", "all"):
        build_ashrae().to_parquet(OUT / "evidence_ashrae.parquet", index=False)
    if a.source in ("lbnl", "all"):
        build_lbnl().to_parquet(OUT / "evidence_lbnl.parquet", index=False)
    if a.source in ("ims", "all"):
        s, summ = build_ims(a.tests)
        if len(s):
            # Merge with earlier runs so tests can be built as their archives finish extracting.
            for name, new in (("evidence_ims_series", s), ("evidence_ims_summary", summ)):
                old = OUT / f"{name}.parquet"
                if a.tests and old.exists():
                    prev = pd.read_parquet(old)
                    new = pd.concat([prev[~prev.test.isin(new.test.unique())], new])
                    new = new.sort_values("test").reset_index(drop=True)
                new.to_parquet(old, index=False)
            s = summ = pd.DataFrame()
        if len(s):
            s.to_parquet(OUT / "evidence_ims_series.parquet", index=False)
            summ.to_parquet(OUT / "evidence_ims_summary.parquet", index=False)


if __name__ == "__main__":
    main()
