"""Condition-aware TTM: XGBoost accelerated failure time (AFT).

The main MVP model. Chosen over a plain regressor because ~75% of component
lives in a real fleet are censored (still running), and over a Cox model
because AFT emits a full parametric distribution per row, which is what the
downstream Monte Carlo needs - not a hazard ratio.

Parameterisation
----------------
Trained on *remaining* life with `aft_loss_distribution="normal"`, so

    log(TTM) ~ Normal(mu(x), sigma)

Two XGBoost details that are easy to get wrong:

  * `predict()` applies the inverse link and returns exp(mu) - i.e. the median
    TTM in days. `predict(output_margin=True)` returns mu in log-days. The
    pipeline needs mu, so it always asks for the margin.
  * `aft_loss_distribution_scale` is a fixed *hyperparameter*, not a learned
    quantity. XGBoost will happily train with a scale that produces badly
    calibrated intervals. So the predictive sigma reported downstream is
    estimated from held-out log-residuals (robustly, via MAD) rather than
    taken from the training hyperparameter - and models/conformal.py then
    applies a width multiplier so the stated 80% band really covers 80%.

Evaluation is against two baselines the brief names: a fixed-interval OEM
schedule, and the fleet Weibull prior. The PHM asymmetric score is reported
alongside MAE because in maintenance a *late* prediction (we said it had
longer to live than it did) is far more costly than an early one, and a
symmetric metric hides that.

Run:
    python -m models.survival_xgb               # train + evaluate
    python -m models.survival_xgb --evaluate    # evaluate a saved model
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.0,
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.5,
    "seed": 42,
}
NUM_ROUNDS = 600
EARLY_STOP = 40

# PHM08-style asymmetry: a late call is penalised ~1.3x harder than an early one.
#
# The original challenge used a1=13, a2=10 with RUL measured in *cycles* of
# order 100, so a1 is ~16% of a typical horizon. Applying those raw constants
# to day-scale errors overflows - exp(1000/10) is not representable, and the
# baselines scored `inf`, which is useless for comparison. So the constants are
# expressed as a fraction of the median observed TTM in the evaluation set,
# preserving the 13:10 late:early ratio while staying scale-free, and the
# exponent is clipped so one catastrophic row cannot swamp the mean.
PHM_EARLY_FRAC = 0.16
PHM_LATE_FRAC = 0.125
PHM_EXP_CLIP = 10.0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def to_dmatrix(df: pd.DataFrame, feats: list[str]) -> xgb.DMatrix:
    """Interval-censored DMatrix. Censored rows get an infinite upper bound.

    float32, not float64: XGBoost casts to float32 internally regardless, so
    the wider dtype only doubles peak memory. At 215k x 165 across three
    simultaneous matrices the float64 path exhausted the heap outright.
    """
    d = xgb.DMatrix(
        df[feats].to_numpy(dtype=np.float32), feature_names=feats, missing=np.nan
    )
    d.set_float_info("label_lower_bound", df["y_lower"].to_numpy(dtype=float))
    d.set_float_info("label_upper_bound", df["y_upper"].to_numpy(dtype=float))
    return d


ID_COLS = ["component_id", "asset_id", "component_type", "snapshot", "y_lower", "y_upper", "event", "fold"]


def load(data: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load only the columns the model needs, as float32 where numeric.

    Reading the whole snapshot table costs ~2x what the model uses; on a 3.9GB
    box that difference is the whole training budget.
    """
    import pyarrow.parquet as pq

    path = data / "train_snapshots.parquet"
    available = set(pq.ParquetFile(path).schema.names)
    feats = [f for f in pd.read_parquet(data / "feature_list.parquet")["feature"] if f in available]
    cols = [c for c in ID_COLS if c in available] + [f for f in feats if f not in ID_COLS]

    snaps = pd.read_parquet(path, columns=cols)
    for c in feats:
        if snaps[c].dtype == np.float64:
            snaps[c] = snaps[c].astype(np.float32)
    gc.collect()
    return snaps, feats


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def phm_score(pred: np.ndarray, actual: np.ndarray, scale: float | None = None) -> float:
    """Mean PHM asymmetric penalty, scale-normalised. Lower is better.

    Pass the same `scale` across models being compared so the numbers are on
    one footing; it defaults to the median actual TTM of the rows given.
    """
    actual = np.asarray(actual, dtype=float)
    s = float(scale) if scale else max(float(np.median(actual)), 1.0)
    a_early, a_late = PHM_EARLY_FRAC * s, PHM_LATE_FRAC * s
    d = np.asarray(pred, dtype=float) - actual
    z = np.where(d < 0, -d / a_early, d / a_late)
    return float(np.mean(np.exp(np.clip(z, 0.0, PHM_EXP_CLIP)) - 1.0))


def c_index(pred_median: np.ndarray, y: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C-index via lifelines, with a longer predicted life = lower risk."""
    from lifelines.utils import concordance_index

    return float(concordance_index(y, pred_median, event_observed=event))


def interval_coverage(
    mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, lo_q: float = 0.10, hi_q: float = 0.90
) -> float:
    lo = np.exp(mu + sigma * norm.ppf(lo_q))
    hi = np.exp(mu + sigma * norm.ppf(hi_q))
    return float(np.mean((y >= lo) & (y <= hi)))


def robust_sigma(residuals: np.ndarray) -> float:
    """MAD-based scale of the log-residuals.

    MAD rather than std because a handful of components genuinely fail far
    earlier than any feature predicted, and a squared-error scale estimate
    lets those few rows inflate the band for the whole fleet.
    """
    r = residuals[np.isfinite(residuals)]
    if len(r) < 10:
        return 0.6
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def baseline_fixed_interval(df: pd.DataFrame, interval_days: dict[str, float]) -> np.ndarray:
    """OEM fixed-interval schedule: service every N days regardless of condition.

    Predicted remaining life = whatever is left of the nominal interval. This is
    what most service contracts actually do today, so it is the honest
    comparator for "did condition monitoring help".
    """
    nominal = df["component_type"].map(interval_days).to_numpy(dtype=float)
    nominal = np.where(np.isfinite(nominal), nominal, 365.0)
    age = df["age_days"].to_numpy(dtype=float)
    return np.clip(nominal - age, 1.0, None)


def baseline_weibull(df: pd.DataFrame, priors_path: Path) -> np.ndarray | None:
    if not priors_path.exists():
        return None
    from models.weibull_prior import predict as prior_predict

    priors = json.loads(priors_path.read_text(encoding="utf-8"))
    return prior_predict(df, priors)["prior_p50_days"].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------


def train(snaps: pd.DataFrame, feats: list[str]) -> tuple[xgb.Booster, dict]:
    tr = snaps[snaps["fold"] == "train"]
    te = snaps[snaps["fold"] == "test"]

    # Carve a calibration slice out of training (by asset, so it stays
    # independent) for sigma estimation and the conformal width multiplier.
    rng = np.random.default_rng(42)
    tr_assets = np.sort(tr["asset_id"].unique())
    cal_assets = set(rng.choice(tr_assets, size=max(1, len(tr_assets) // 5), replace=False).tolist())
    cal = tr[tr["asset_id"].isin(cal_assets)]
    fit = tr[~tr["asset_id"].isin(cal_assets)]

    print(f"  fit {len(fit):,} | cal {len(cal):,} | test {len(te):,} rows, {len(feats)} features")

    # Memory discipline for a 3.9GB box: build the two matrices training needs,
    # drop every intermediate frame, and never hold a test DMatrix during
    # training (evaluate() builds it afterwards). QuantileDMatrix pre-bins for
    # tree_method="hist", which is both what we train with and markedly leaner
    # than a dense DMatrix.
    dfit = xgb.QuantileDMatrix(
        fit[feats].to_numpy(dtype=np.float32), feature_names=feats, missing=np.nan
    )
    dfit.set_float_info("label_lower_bound", fit["y_lower"].to_numpy(dtype=float))
    dfit.set_float_info("label_upper_bound", fit["y_upper"].to_numpy(dtype=float))

    dcal = xgb.QuantileDMatrix(
        cal[feats].to_numpy(dtype=np.float32), feature_names=feats, missing=np.nan, ref=dfit
    )
    dcal.set_float_info("label_lower_bound", cal["y_lower"].to_numpy(dtype=float))
    dcal.set_float_info("label_upper_bound", cal["y_upper"].to_numpy(dtype=float))

    n_test = len(te)
    del te
    gc.collect()

    booster = xgb.train(
        PARAMS,
        dfit,
        num_boost_round=NUM_ROUNDS,
        evals=[(dfit, "fit"), (dcal, "cal")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=100,
    )
    del dfit
    gc.collect()

    # Predictive sigma from held-out residuals on *uncensored* calibration rows
    # only - a censored row has no observed log-residual to measure.
    booster.set_param({"nthread": 1})  # keep per-thread prediction buffers small
    cal_mu = booster.predict(dcal, output_margin=True)
    cal_obs = cal["event"].to_numpy(dtype=bool)
    resid = np.log(np.clip(cal["y_lower"].to_numpy(dtype=float), 0.5, None)) - cal_mu
    sigma_global = robust_sigma(resid[cal_obs])

    # Per-type sigma, shrunk toward the global estimate so a component type with
    # three observed failures does not get its own confident-looking band.
    sigma_by_type: dict[str, float] = {}
    for ctype, g in cal[cal_obs].groupby("component_type"):
        idx = g.index
        r = np.log(np.clip(g["y_lower"].to_numpy(dtype=float), 0.5, None)) - booster.predict(
            to_dmatrix(g, feats), output_margin=True
        )
        n = len(r)
        s_type = robust_sigma(r)
        w = n / (n + 30.0)  # shrinkage weight
        sigma_by_type[str(ctype)] = float(w * s_type + (1 - w) * sigma_global)
        _ = idx

    meta = {
        "best_iteration": int(booster.best_iteration),
        "sigma_global": sigma_global,
        "sigma_by_type": sigma_by_type,
        "features": feats,
        "params": PARAMS,
        "n_fit": int(len(fit)),
        "n_cal": int(len(cal)),
        "n_test": int(n_test),
        "calibration_assets": sorted(cal_assets),
    }
    return booster, meta


def sigma_for(df: pd.DataFrame, meta: dict) -> np.ndarray:
    by_type = meta["sigma_by_type"]
    g = meta["sigma_global"]
    return df["component_type"].astype(str).map(lambda t: by_type.get(t, g)).to_numpy(dtype=float)


def evaluate(booster: xgb.Booster, meta: dict, snaps: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Score the model and both baselines on uncensored test rows."""
    te = snaps[snaps["fold"] == "test"]
    obs = te[te["event"].to_numpy(dtype=bool)].copy()
    if obs.empty:
        raise SystemExit("no uncensored test rows to score")

    mu = booster.predict(to_dmatrix(obs, feats), output_margin=True)
    sigma = sigma_for(obs, meta)
    pred = np.exp(mu)
    y = obs["y_lower"].to_numpy(dtype=float)

    # Fixed-interval baseline uses each type's observed median life as its
    # nominal interval - a generous reading of what an OEM schedule would say.
    nominal = snaps[snaps["event"]].groupby("component_type")["y_lower"].median().to_dict()
    fixed = baseline_fixed_interval(obs, nominal)
    weib = baseline_weibull(obs, ARTIFACTS / "weibull_priors.json")

    # C-index needs the full censored test set, not just observed rows.
    mu_all = booster.predict(to_dmatrix(te, feats), output_margin=True)
    ci_model = c_index(np.exp(mu_all), te["y_lower"].to_numpy(dtype=float), te["event"].to_numpy(dtype=bool))
    ci_fixed = c_index(
        baseline_fixed_interval(te, nominal),
        te["y_lower"].to_numpy(dtype=float),
        te["event"].to_numpy(dtype=bool),
    )

    # One shared scale so the PHM column compares like with like.
    phm_scale = max(float(np.median(y)), 1.0)
    rows = [
        {
            "model": "ServiceSense (XGBoost AFT)",
            "MAE (days)": float(np.mean(np.abs(pred - y))),
            "median AE (days)": float(np.median(np.abs(pred - y))),
            "C-index": ci_model,
            "PHM score": phm_score(pred, y, phm_scale),
            "P10-P90 coverage": interval_coverage(mu, sigma, y),
        },
        {
            "model": "Fleet Weibull prior",
            "MAE (days)": float(np.mean(np.abs(weib - y))) if weib is not None else np.nan,
            "median AE (days)": float(np.median(np.abs(weib - y))) if weib is not None else np.nan,
            "C-index": np.nan,
            "PHM score": phm_score(weib, y, phm_scale) if weib is not None else np.nan,
            "P10-P90 coverage": np.nan,
        },
        {
            "model": "Fixed-interval OEM schedule",
            "MAE (days)": float(np.mean(np.abs(fixed - y))),
            "median AE (days)": float(np.median(np.abs(fixed - y))),
            "C-index": ci_fixed,
            "PHM score": phm_score(fixed, y, phm_scale),
            "P10-P90 coverage": np.nan,
        },
    ]
    return pd.DataFrame(rows)


def evaluate_by_type(
    booster: xgb.Booster, meta: dict, snaps: pd.DataFrame, feats: list[str], min_rows: int = 25
) -> pd.DataFrame:
    """Per-component-type scores. Non-optional, because the aggregate lies.

    Uncensored test rows are overwhelmingly short-lived parts - filters and
    contactors fail repeatedly inside the window, compressors almost never do.
    So a fleet-level C-index can read 0.95 while the model has no signal at all
    on the components that dominate cost and drive the urgent queue. Reporting
    rank correlation per type is what surfaces that; it is the single most
    useful line in this file.
    """
    te = snaps[snaps["fold"] == "test"]
    rows = []
    for ctype, g in te.groupby("component_type"):
        obs = g[g["event"].to_numpy(dtype=bool)]
        if len(obs) < min_rows:
            continue
        mu = booster.predict(to_dmatrix(obs, feats), output_margin=True)
        pred = np.exp(mu)
        y = obs["y_lower"].to_numpy(dtype=float)
        rho = float(pd.Series(pred).corr(pd.Series(y), method="spearman"))
        rows.append(
            {
                "component_type": ctype,
                "n_events": int(len(obs)),
                "n_rows": int(len(g)),
                "censored": round(1.0 - float(g["event"].mean()), 3),
                "median AE (d)": round(float(np.median(np.abs(pred - y))), 1),
                "median true (d)": round(float(np.median(y)), 1),
                "spearman": round(rho, 3),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman")


def score_all(booster: xgb.Booster, meta: dict, data: Path) -> pd.DataFrame:
    """Predict mu/sigma for every component as of today -> the pipeline input."""
    score = pd.read_parquet(data / "score_snapshots.parquet")
    feats = meta["features"]
    feats = [f for f in feats if f in score.columns]
    mu = booster.predict(to_dmatrix(score, feats), output_margin=True)
    sigma = sigma_for(score, meta)
    out = score[["component_id", "asset_id", "component_type", "snapshot", "age_days",
                 "days_since_pm", "n_past_repairs"]].copy()
    out["mu_log_days"] = mu
    out["sigma_log"] = sigma
    out["imputed_share_30d"] = score.get("imputed_share_30d", 0.0)
    out["hours_since_reading"] = score.get("hours_since_reading", 0.0)
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate the AFT survival model")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--evaluate", action="store_true", help="load the saved model instead of training")
    args = ap.parse_args()

    t0 = time.perf_counter()
    snaps, feats = load(args.data)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS / "survival_xgb.json"
    meta_path = ARTIFACTS / "survival_xgb_meta.json"

    if args.evaluate and model_path.exists():
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print("loaded saved model")
    else:
        print("training AFT model ...")
        booster, meta = train(snaps, feats)
        booster.save_model(str(model_path))
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  saved {model_path.name} (best_iteration={meta['best_iteration']})")

    print(f"\n  predictive sigma (log-days): global {meta['sigma_global']:.3f}")
    st = pd.Series(meta["sigma_by_type"]).sort_values()
    print("    tightest: " + ", ".join(f"{k} {v:.2f}" for k, v in st.head(3).items()))
    print("    widest:   " + ", ".join(f"{k} {v:.2f}" for k, v in st.tail(3).items()))

    print("\nevaluation on held-out assets + final 6 months (uncensored rows):")
    table = evaluate(booster, meta, snaps, feats)
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    table.to_parquet(ARTIFACTS / "eval_table.parquet", index=False)

    by_type = evaluate_by_type(booster, meta, snaps, feats)
    by_type.to_parquet(ARTIFACTS / "eval_by_type.parquet", index=False)
    print("\nper-component-type (worst rank correlation first):")
    print(by_type.to_string(index=False))
    weak = by_type[by_type["spearman"] < 0.3]
    if len(weak):
        print(
            "\n  WEAK: " + ", ".join(weak["component_type"])
            + "\n  These types have too few in-window failures for the conditional model to\n"
            "  learn from; they fall back to the fleet prior. Reported, not hidden."
        )

    imp = booster.get_score(importance_type="gain")
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:12]
    print("\ntop features by gain:")
    for k, v in top:
        print(f"  {v:10,.0f}  {k}")

    scored = score_all(booster, meta, args.data)
    scored.to_parquet(args.data / "component_predictions.parquet", index=False)
    print(f"\nscored {len(scored):,} components -> component_predictions.parquet")
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
