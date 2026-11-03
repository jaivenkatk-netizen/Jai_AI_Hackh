"""FastAPI over the precomputed DuckDB.

Every endpoint is a thin SELECT against a view built by pipeline/build_all.py.
Nothing here trains, samples or optimises - that all happened offline, which is
what keeps the demo fast and means a model bug is a pipeline re-run rather than
a debugging session inside a web request.

Two exceptions, both deliberate:
  * /api/whatif re-runs the twin -> TTM -> parts chain for a *single* asset,
    which is the one place the audience should see the model think.
  * /ws/telemetry replays stored hourly telemetry at ~60x so the Command Center
    looks live without anything actually being live.

DuckDB threading: one process-wide read-only connection, with a `.cursor()` per
request. A DuckDB connection is not safe to use concurrently, but cursors from
it are independent, so this gives thread-safe reads without a lock or a pool.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os

# Cap BLAS threading *before* numpy/scipy are imported anywhere in the process.
# OpenBLAS reserves sizeable per-thread scratch buffers at init; on this 3.9GB
# box the default thread count exhausted the heap and the first /api/asset
# request died with "OpenBLAS error: Memory allocation still failed after 10
# retries". The API does no heavy linear algebra - every endpoint is a SELECT -
# so single-threaded BLAS costs nothing here.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import asyncio  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from datetime import date, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.whatif import PRESETS, WhatIfEngine, run_whatif  # noqa: E402
from contracts.schemas import Bucket, WhatIfRequest, WhatIfResponse

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("SERVICESENSE_DB", ROOT / "data" / "servicesense.duckdb"))

# Replay speed for the telemetry WebSocket: one stored hour per (3600/60) ms.
REPLAY_SPEED = float(os.environ.get("SERVICESENSE_REPLAY_SPEED", "60"))

_con: duckdb.DuckDBPyConnection | None = None
_whatif: WhatIfEngine | None = None


def db() -> duckdb.DuckDBPyConnection:
    if _con is None:
        raise HTTPException(503, "database not open")
    return _con.cursor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _con
    if not DB_PATH.exists():
        raise RuntimeError(
            f"{DB_PATH} not found - run `python -m pipeline.build_all` first"
        )
    _con = duckdb.connect(str(DB_PATH), read_only=True)
    global _whatif
    # Load the booster once at startup, not per request: a cold load is ~900 ms
    # and the what-if slider is meant to feel immediate.
    try:
        _whatif = WhatIfEngine()
        # Warm the whole path, not just the booster: the first run also pays
        # CoolProp's fluid load, scipy's solver import and DuckDB's first
        # touch of the repair-kit tables. Doing it here costs ~1.7 s at
        # startup and makes the first slider drag as fast as the rest.
        warm_id = _con.execute("SELECT asset_id FROM score_snapshots LIMIT 1").fetchone()
        if warm_id:
            run_whatif(_con.cursor(), _whatif, warm_id[0], d_ambient=1.0)
    except Exception as exc:  # the rest of the API is still fully usable
        print(f"what-if unavailable ({exc}); other endpoints unaffected")
        _whatif = None
    yield
    _con.close()
    _con = None
    _whatif = None


app = FastAPI(title="ServiceSense API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _jsonish(v: Any) -> Any:
    """Parse the JSON-encoded text columns back into structures.

    Assumption cards, evidence bullets and confidence reasons are stored as JSON
    strings in Parquet/DuckDB because they are variable-length lists of objects;
    the API hands them back as real JSON rather than making the client parse
    strings out of a payload.
    """
    if v is None or isinstance(v, (list, dict)):
        return v if v is not None else []
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return [v]
    return []


def _clean(v: Any) -> Any:
    """Make a DuckDB/pandas value JSON-safe (NaN -> None, numpy -> python).

    List-valued columns are handled first and explicitly. DuckDB hands LIST
    columns back as numpy arrays, and any `v or []` / `pd.isna(v)` on one raises
    "truth value of an array is ambiguous" - which is exactly how the first
    /api/asset call 500'd. Centralising it here means no call site has to
    remember.
    """
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "tolist") and getattr(v, "ndim", 0) > 0:
        return [_clean(x) for x in v.tolist()]
    if v is None:
        return None
    if isinstance(v, (datetime, pd.Timestamp)):
        return pd.Timestamp(v).isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, float) and (pd.isna(v) or v in (float("inf"), float("-inf"))):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


def rows(df: pd.DataFrame, json_cols: tuple[str, ...] = ()) -> list[dict]:
    out: list[dict] = []
    for rec in df.to_dict("records"):
        d = {k: (_jsonish(v) if k in json_cols else _clean(v)) for k, v in rec.items()}
        out.append(d)
    return out


def _as_of(con: duckdb.DuckDBPyConnection) -> str:
    v = con.execute("SELECT MAX(ts) FROM telemetry_daily").fetchone()[0]
    return pd.Timestamp(v).date().isoformat()


# --------------------------------------------------------------------------
# GET /api/kpis  -> Command Center
# --------------------------------------------------------------------------


@app.get("/api/kpis")
def get_kpis(hub: str | None = Query(None)) -> dict:
    con = db()
    where = "WHERE hub_id = ?" if hub else ""
    args = [hub] if hub else []

    agg = con.execute(
        f"""
        SELECT
            COUNT(DISTINCT CASE WHEN bucket <> 'forecast' THEN asset_id END) AS assets_at_risk,
            COUNT(DISTINCT CASE WHEN bucket = 'urgent' THEN asset_id END) AS urgent_assets
        FROM v_asset_risk {where}
        """,
        args,
    ).fetchone()

    jobs = con.execute(
        f"SELECT COUNT(*) FROM v_urgent_jobs {where}", args
    ).fetchone()[0]

    stock = con.execute(
        f"""
        SELECT
            SUM(CASE WHEN shortfall_vs_p90 > 0 THEN 1 ELSE 0 END) AS projected_stockouts,
            SUM(excess_value) AS excess_value
        FROM stock_positions {where}
        """,
        args,
    ).fetchone()

    recs = con.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN avoided_action = 'expedite' THEN 1 ELSE 0 END) AS expedites_avoided,
            SUM(CASE WHEN action = 'transfer' THEN 1 ELSE 0 END) AS transfers
        FROM recommendations {'WHERE to_hub_id = ?' if hub else ''}
        """,
        args,
    ).fetchone()

    fresh = con.execute(
        "SELECT date_diff('hour', MAX(ts), (SELECT MAX(ts) FROM telemetry_hourly)) "
        "FROM telemetry_hourly"
    ).fetchone()[0]

    tiles = [
        {
            "key": "assets_at_risk", "label": "Assets at risk",
            "value": int(agg[0] or 0), "unit": "assets", "tone": "warning",
            "drill_filter": {"bucket": "planned"},
        },
        {
            "key": "urgent_jobs", "label": "Urgent jobs (0-14 d)",
            "value": int(jobs or 0), "unit": "jobs", "tone": "critical",
            "drill_filter": {"bucket": "urgent"},
        },
        {
            "key": "projected_stockouts", "label": "Projected stockouts",
            "value": int(stock[0] or 0), "unit": "part-hubs", "tone": "critical",
            "drill_filter": {"shortage": True},
        },
        {
            "key": "expedites_avoided", "label": "Expedites avoided",
            "value": int(recs[1] or 0), "unit": "orders", "tone": "good",
            "drill_filter": {"action": "transfer"},
        },
        {
            "key": "excess_value", "label": "Excess stock value",
            "value": float(stock[1] or 0), "unit": "INR", "tone": "neutral",
            "drill_filter": {"excess": True},
        },
    ]

    hubs = con.execute("SELECT * FROM v_hub_risk ORDER BY urgent_jobs DESC").df()
    # Normalise risk to the busiest hub so the map colour scale is stable
    mx = max(float(hubs["urgent_jobs"].max() or 1), 1.0)
    hub_rows = rows(hubs)
    for h in hub_rows:
        h["risk_score"] = round(min((h.get("urgent_jobs") or 0) / mx, 1.0), 3)

    queue = con.execute(
        f"""
        SELECT component_id, asset_id, asset_label, site, hub_id, component_type,
               p10_days, p50_days, p90_days, p_fail_14d, confidence,
               confidence_reasons, assumptions, parts_short
        FROM v_urgent_jobs {where}
        ORDER BY p_fail_14d DESC, p50_days ASC
        LIMIT 25
        """,
        args,
    ).df()

    return {
        "generated_at": datetime.now().isoformat(),
        "as_of_date": _as_of(con),
        "tiles": tiles,
        "hubs": hub_rows,
        "urgent_queue": rows(queue, json_cols=("confidence_reasons", "assumptions")),
        "data_freshness_hours": float(fresh or 0),
        "totals": {
            "recommendations": int(recs[0] or 0),
            "transfers": int(recs[2] or 0),
        },
    }


# --------------------------------------------------------------------------
# GET /api/fleet  -> Fleet Health
# --------------------------------------------------------------------------


@app.get("/api/fleet")
def get_fleet(
    hub: str | None = Query(None),
    bucket: Bucket | None = Query(None),
    limit: int = Query(300, le=2000),
) -> dict:
    con = db()
    clauses, args = [], []
    if hub:
        clauses.append("hub_id = ?")
        args.append(hub)
    if bucket:
        clauses.append("bucket = ?")
        args.append(bucket.value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    total = con.execute(f"SELECT COUNT(*) FROM v_asset_risk {where}", args).fetchone()[0]
    df = con.execute(
        f"""
        SELECT * FROM v_asset_risk {where}
        ORDER BY p_fail_14d DESC, p50_days ASC
        LIMIT {int(limit)}
        """,
        args,
    ).df()

    heat = con.execute(
        """
        SELECT region, component_type, AVG(p_fail_90d) AS risk
        FROM v_component_risk
        GROUP BY region, component_type
        """
    ).df()
    heatmap: dict[str, dict[str, float]] = {}
    for r in heat.itertuples(index=False):
        heatmap.setdefault(str(r.region), {})[str(r.component_type)] = round(float(r.risk), 4)

    return {
        "as_of_date": _as_of(con),
        "total": int(total),
        "rows": rows(df, json_cols=("confidence_reasons",)),
        "risk_heatmap": heatmap,
    }


# --------------------------------------------------------------------------
# GET /api/asset/{asset_id}  -> Asset Twin
# --------------------------------------------------------------------------

TREND_SIGNALS = ("discharge_p", "suction_p", "superheat", "comp_current", "vib_rms", "cop")
SIGNAL_META = {
    "discharge_p": ("Discharge pressure", "kPa"),
    "suction_p": ("Suction pressure", "kPa"),
    "superheat": ("Superheat", "K"),
    "comp_current": ("Compressor current", "A"),
    "vib_rms": ("Vibration RMS", "mm/s"),
    "cop": ("COP", "-"),
}


@app.get("/api/asset/{asset_id}")
def get_asset(asset_id: str, trend_days: int = Query(90, le=365)) -> dict:
    con = db()
    a = con.execute(
        """
        SELECT asset_id, asset_label, asset_type, model, refrigerant, site_name, hub_id,
               region, climate_zone, capacity_kw, size_class, install_date, criticality
        FROM assets WHERE asset_id = ?
        """,
        [asset_id],
    ).df()
    if a.empty:
        raise HTTPException(404, f"asset {asset_id} not found")
    meta = rows(a)[0]

    comps = con.execute(
        """
        SELECT t.*, COALESCE(v.bucket, 'forecast') AS bucket
        FROM component_ttm t
        LEFT JOIN v_component_risk v ON v.component_id = t.component_id
        WHERE t.asset_id = ?
        ORDER BY t.p50_days ASC
        """,
        [asset_id],
    ).df()

    drivers = con.execute(
        """
        SELECT component_id, rank, feature, display_name, shap_value, value, direction
        FROM component_drivers
        WHERE component_id IN (SELECT component_id FROM component_ttm WHERE asset_id = ?)
        ORDER BY component_id, rank
        """,
        [asset_id],
    ).df()
    by_comp: dict[str, list[dict]] = {}
    for d in rows(drivers):
        by_comp.setdefault(str(d["component_id"]), []).append(d)

    components = []
    for c in rows(comps, json_cols=("confidence_reasons", "assumptions")):
        c["drivers"] = by_comp.get(str(c["component_id"]), [])
        c["survival"] = _survival_curve(float(c["mu_log_days"]), float(c["sigma_effective"]))
        components.append(c)

    # Sensor trends with the twin's healthy baseline. Until sim/asset_twin.py
    # lands (Day 2 PM) the baseline is the asset's own early-window behaviour
    # at comparable ambient, which is a defensible stand-in and is labelled as
    # such in the payload rather than presented as physics.
    tel = con.execute(
        f"""
        SELECT ts, {', '.join(TREND_SIGNALS)}, ambient_t_max
        FROM telemetry_daily
        WHERE asset_id = ?
          AND ts >= (SELECT MAX(ts) - INTERVAL {int(trend_days)} DAY FROM telemetry_daily)
        ORDER BY ts
        """,
        [asset_id],
    ).df()
    base = con.execute(
        f"""
        SELECT {', '.join(f'MEDIAN({s}) AS {s}' for s in TREND_SIGNALS)}
        FROM telemetry_daily
        WHERE asset_id = ?
          AND ts <= (SELECT MIN(ts) + INTERVAL 180 DAY FROM telemetry_daily)
        """,
        [asset_id],
    ).fetchone()

    sensors = []
    for i, sig in enumerate(TREND_SIGNALS):
        if sig not in tel.columns or tel[sig].isna().all():
            continue
        label, unit = SIGNAL_META[sig]
        b = _clean(base[i]) if base else None
        observed = [_clean(v) for v in tel[sig].tolist()]
        baseline = [b] * len(observed) if b is not None else []
        sensors.append(
            {
                "signal": sig,
                "display_name": label,
                "unit": unit,
                "timestamps": [pd.Timestamp(t).isoformat() for t in tel["ts"]],
                "observed": observed,
                "baseline": baseline,
                "baseline_source": "early-window median (physics twin pending)",
                "residual": [
                    None if (o is None or b is None) else round(o - b, 4) for o in observed
                ],
            }
        )

    history = con.execute(
        """
        SELECT wo_id, open_ts AS ts, type, component_type, failure_mode,
               parts_used, first_time_fix, notes, downtime_h
        FROM work_orders WHERE asset_id = ?
        ORDER BY open_ts DESC LIMIT 20
        """,
        [asset_id],
    ).df()
    hist = rows(history)

    return {
        **meta,
        "twin": _twin_state(tel, comps, meta),
        "fault_state": _fault_state(asset_id),
        "sensors": sensors,
        "components": components,
        "history": hist,
    }


def _survival_curve(mu: float, sigma: float, n: int = 60) -> dict:
    """S(t) = P(TTM > t) read off the fitted log-normal.

    scipy/math are imported at module scope rather than here: importing scipy
    inside a request handler pulls OpenBLAS into the request path, which is
    both slow on first call and how the original 500 happened.
    """
    hi = float(math.exp(mu + 2.5 * max(sigma, 1e-6)))
    days = [round(hi * i / n, 2) for i in range(1, n + 1)]
    surv = [
        round(float(1.0 - norm.cdf((math.log(max(d, 1e-9)) - mu) / max(sigma, 1e-9))), 5)
        for d in days
    ]
    return {"days": days, "survival": surv}


def _twin_state(tel: pd.DataFrame, comps: pd.DataFrame, meta: dict) -> dict:
    """Current cycle state plus per-component health for the schematic."""
    last = tel.tail(1)
    g = lambda c: _clean(last[c].iloc[0]) if (len(last) and c in last.columns) else None  # noqa: E731
    # Health index from the predicted band: a component deep into its life has
    # a short P50 relative to its own type, so rank within the asset is the
    # honest signal available before the physics twin lands.
    health: dict[str, float] = {}
    for c in comps.to_dict("records"):
        p50 = float(c.get("p50_days") or 0)
        health[str(c["component_type"])] = round(min(max(p50 / 1825.0, 0.02), 1.0), 3)
    return {
        "refrigerant": meta.get("refrigerant"),
        "suction_p_kpa": g("suction_p"),
        "discharge_p_kpa": g("discharge_p"),
        "superheat_k": g("superheat"),
        "cop": g("cop"),
        "ambient_t_max": g("ambient_t_max"),
        "component_health": health,
        "source": "observed telemetry",
    }


FAULT_STATE_PARQUET = ROOT / "data" / "synthetic" / "fault_state.parquet"


def _fault_state(asset_id: str) -> dict | None:
    """Physics-informed inferred fault state for one asset.

    Read straight from the parquet rather than a DuckDB table so it can be
    refreshed by `python -m models.pinn --infer` without rebuilding the whole
    pipeline. Returns None when the file is absent, so the screen degrades to
    telemetry-only rather than erroring.

    Only `charge_frac` survived validation (r = 0.786 against the generator's
    true refrigerant-circuit damage). The other two parameters are returned
    with `validated: false` and must be rendered as such - they saturate
    against their bounds and track ambient rather than damage.
    """
    if not FAULT_STATE_PARQUET.exists():
        return None
    con = db()
    try:
        df = con.execute(
            "SELECT * FROM read_parquet(?) WHERE asset_id = ?",
            [str(FAULT_STATE_PARQUET), asset_id],
        ).df()
    except Exception:
        return None
    if df.empty:
        return None
    r = df.iloc[0]
    params = []
    for key, label in (
        ("charge_frac", "Refrigerant charge"),
        ("cond_deg", "Condenser degradation"),
        ("evap_deg", "Evaporator degradation"),
    ):
        if key not in df.columns:
            continue
        params.append(
            {
                "key": key,
                "label": label,
                "value": _clean(r[key]),
                "validated": bool(r.get(f"{key}_validated", False)),
            }
        )
    return {
        "params": params,
        "physics_residual": _clean(r.get("physics_residual")),
        "explains_sensors": bool(r.get("explains_sensors", False)),
        "as_of": str(r["ts"])[:10],
        "method": (
            "Inverse network trained on the twin's physics, no labels. "
            "Validated against known damage: charge r=0.79; the condenser and "
            "evaporator parameters did not survive validation."
        ),
    }


# --------------------------------------------------------------------------
# GET /api/parts-demand  -> Parts Demand
# --------------------------------------------------------------------------


@app.get("/api/parts-demand")
def get_parts_demand(
    hub: str | None = Query(None),
    bucket: Bucket | None = Query(None),
    limit: int = Query(400, le=2000),
) -> dict:
    con = db()
    clauses, args = [], []
    if hub:
        clauses.append("d.hub_id = ?")
        args.append(hub)
    if bucket:
        clauses.append("d.bucket = ?")
        args.append(bucket.value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    df = con.execute(
        f"""
        SELECT d.*, s.on_hand, s.on_order, s.reserved, s.available,
               s.reorder_point, s.safety_stock, s.shortfall_vs_p90
        FROM parts_demand d
        LEFT JOIN stock_positions s ON s.part_no = d.part_no AND s.hub_id = d.hub_id
        {where}
        ORDER BY CASE d.bucket WHEN 'urgent' THEN 0 WHEN 'planned' THEN 1 ELSE 2 END,
                 d.total_p90 DESC
        LIMIT {int(limit)}
        """,
        args,
    ).df()

    # Traceability: the assets and failure modes behind each urgent line
    contrib = con.execute(
        f"""
        SELECT part_no, hub_id, asset_id, component_id, component_type,
               failure_mode, p_contribution
        FROM parts_demand_contrib
        {'WHERE hub_id = ?' if hub else ''}
        ORDER BY p_contribution DESC
        """,
        [hub] if hub else [],
    ).df()
    by_key: dict[tuple[str, str], list[dict]] = {}
    for c in rows(contrib):
        by_key.setdefault((str(c["part_no"]), str(c["hub_id"])), []).append(c)

    out = []
    for r in rows(df):
        key = (str(r["part_no"]), str(r["hub_id"]))
        cs = by_key.get(key, [])[:8]
        r["contributing"] = cs
        r["contributing_assets"] = sorted({c["asset_id"] for c in cs})
        r["contributing_failure_modes"] = sorted({c["failure_mode"] for c in cs})
        out.append(r)

    kits = con.execute(
        """
        SELECT component_type, failure_mode, size_class, refrigerant, part_no,
               description, qty_mean AS qty, p_used, reason, unit_cost,
               expected_first_time_fix
        FROM van_kits
        ORDER BY component_type, failure_mode, p_used DESC
        """
    ).df()

    return {
        "as_of_date": _as_of(con),
        "rows": out,
        "van_kits": rows(kits),
    }


# --------------------------------------------------------------------------
# GET /api/recommendations  -> Inventory Planner
# --------------------------------------------------------------------------


@app.get("/api/recommendations")
def get_recommendations(
    hub: str | None = Query(None),
    action: str | None = Query(None),
    limit: int = Query(200, le=1000),
) -> dict:
    con = db()
    clauses, args = [], []
    if hub:
        clauses.append("(to_hub_id = ? OR from_hub_id = ?)")
        args += [hub, hub]
    if action:
        clauses.append("action = ?")
        args.append(action)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    cards = con.execute(
        f"""
        SELECT * FROM recommendations {where}
        ORDER BY need_in_days ASC, criticality ASC
        LIMIT {int(limit)}
        """,
        args,
    ).df()

    pos = con.execute(
        f"""
        SELECT p.*, pt.description
        FROM stock_positions p
        LEFT JOIN parts pt ON pt.part_no = p.part_no
        {'WHERE p.hub_id = ?' if hub else ''}
        ORDER BY p.shortfall_vs_p90 DESC, p.excess_value DESC
        LIMIT 400
        """,
        [hub] if hub else [],
    ).df()

    totals = con.execute(
        f"""
        SELECT SUM(excess_value) AS excess, COUNT(*) AS positions
        FROM stock_positions {'WHERE hub_id = ?' if hub else ''}
        """,
        [hub] if hub else [],
    ).fetchone()

    avoided_clauses = [*clauses, "avoided_action = 'expedite'"]
    avoided = con.execute(
        f"SELECT COUNT(*) FROM recommendations WHERE {' AND '.join(avoided_clauses)}",
        args,
    ).fetchone()[0]

    return {
        "as_of_date": _as_of(con),
        "cards": rows(cards, json_cols=("evidence", "assumptions")),
        "positions": rows(pos),
        "total_excess_value": float(totals[0] or 0),
        "expedites_avoided": int(avoided or 0),
    }


# --------------------------------------------------------------------------
# GET /api/hubs, /api/meta  -> filters and provenance
# --------------------------------------------------------------------------


@app.get("/api/hubs")
def get_hubs() -> list[dict]:
    return rows(db().execute("SELECT * FROM v_hub_risk ORDER BY name").df())


@app.get("/api/meta")
def get_meta() -> dict:
    """Provenance and model scores - what the Models & Data screen reads."""
    con = db()
    art = ROOT / "models" / "artifacts"

    def _parquet(name: str) -> list[dict]:
        p = art / name
        return rows(pd.read_parquet(p)) if p.exists() else []

    meta_path = art / "survival_xgb_meta.json"
    model_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    counts = con.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM assets) AS assets,
            (SELECT COUNT(*) FROM components) AS components,
            (SELECT COUNT(*) FROM telemetry_daily) AS telemetry_days,
            (SELECT COUNT(*) FROM work_orders) AS work_orders,
            (SELECT COUNT(*) FROM pdm_lives) AS pdm_component_lives
        """
    ).df()

    return {
        "as_of_date": _as_of(con),
        "counts": rows(counts)[0] if len(counts) else {},
        "eval_table": _parquet("eval_table.parquet"),
        "eval_by_type": _parquet("eval_by_type.parquet"),
        "calibration_curve": _parquet("calibration_curve.parquet"),
        "coverage_by_type": _parquet("coverage_by_type.parquet"),
        "model": {
            "sigma_global": model_meta.get("sigma_global"),
            "calibration_multiplier": model_meta.get("calibration_multiplier"),
            "best_iteration": model_meta.get("best_iteration"),
            "n_fit": model_meta.get("n_fit"),
            "n_test": model_meta.get("n_test"),
            "features": len(model_meta.get("features", [])),
        },
        "data_sources": [
            {
                "name": "Synthetic HVAC fleet",
                "role": "Links assets, BOM, hubs, stock and suppliers - no public set does",
                "detail": "500 assets x 5 years, seeded; stress-driven Weibull degradation",
            },
            {
                "name": "Microsoft/Azure Predictive Maintenance",
                "role": "Validates the survival pipeline on real component failure history",
                "detail": "100 machines, 876k telemetry rows, 761 failures, 77% censoring",
            },
        ],
    }


# --------------------------------------------------------------------------
# WS /ws/telemetry  -> live-looking Command Center
# --------------------------------------------------------------------------


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket, speed: float = REPLAY_SPEED) -> None:
    """Replay stored hourly telemetry so the dashboard animates.

    Nothing is live; this is the last 90 days of the 50-asset hourly subset
    played back at `speed`x. Honest framing matters for the pitch - it is a
    replay, not a stream, and the payload says so.
    """
    await ws.accept()
    con = db()
    try:
        df = con.execute(
            """
            SELECT ts, asset_id, discharge_p, suction_p, comp_current, vib_rms, cop, ambient_t
            FROM telemetry_hourly
            WHERE ts >= (SELECT MAX(ts) - INTERVAL 14 DAY FROM telemetry_hourly)
            ORDER BY ts
            """
        ).df()
        if df.empty:
            await ws.close()
            return
        delay = max(3600.0 / max(speed, 1.0) / 1000.0, 0.02)
        for ts, grp in df.groupby("ts", sort=True):
            await ws.send_json(
                {
                    "type": "telemetry",
                    "replay": True,
                    "ts": pd.Timestamp(ts).isoformat(),
                    "readings": rows(grp.drop(columns=["ts"])),
                }
            )
            await asyncio.sleep(delay)
        await ws.send_json({"type": "replay_complete"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # keep a demo alive rather than 500-ing the socket
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


@app.get("/api/whatif/presets")
def whatif_presets() -> dict:
    return {"presets": [{"name": k, **v} for k, v in PRESETS.items()]}


@app.post("/api/whatif")
def post_whatif(req: WhatIfRequest) -> dict:
    """Re-score one asset under shifted operating conditions.

    The only endpoint that runs the model inside a request. Everything else is
    a SELECT against a precomputed table, which is what keeps the demo fast;
    this one is narrow enough (one asset, a dozen components) to afford it.
    """
    if _whatif is None:
        raise HTTPException(503, "what-if engine unavailable - model artifact missing")
    d = {
        "d_ambient": req.d_ambient_c,
        "d_load": req.d_load_frac,
        "d_duty": req.d_duty_hours,
    }
    if req.preset:
        if req.preset not in PRESETS:
            raise HTTPException(422, f"unknown preset {req.preset!r}")
        d = dict(PRESETS[req.preset])
    con = db()
    try:
        out = run_whatif(con, _whatif, req.asset_id, **d)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if req.preset:
        out["preset"] = req.preset
    # Validate against the frozen contract on the way out. This is the one
    # endpoint that builds its payload by hand rather than from a table, so
    # it is the one most able to drift from the schema the UI is typed against.
    return _clean(WhatIfResponse.model_validate(out).model_dump())


@app.get("/api/health")
def health() -> dict:
    con = db()
    n = con.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
    return {"ok": True, "db": str(DB_PATH), "recommendations": int(n)}
