"""Read-only tools the copilot agents call.

Every number the copilot says must trace to a row one of these returned, so the
tools are thin SELECTs over the same DuckDB views the dashboard reads, and they
never write. Each returns:

    {"summary": "<one line for the trace panel>", "rows": [...], "row_count": n}

`summary` is what the UI shows beside the answer; `rows` is what the model (or
the offline synthesiser) reads and what the auditor checks the answer against.

Which agent may call which tool is declared in agents.py - a specialist only
sees the tools that belong to its job.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import duckdb
import pandas as pd

MAX_ROWS = 25


def _clean(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        return _clean(v.tolist())
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 3)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _rows(df: pd.DataFrame, limit: int = MAX_ROWS) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.head(limit).to_dict("records")]


class ToolBox:
    """Tool implementations bound to one DuckDB cursor (and optionally the what-if engine)."""

    def __init__(self, con: duckdb.DuckDBPyConnection, whatif: Callable[..., dict] | None = None):
        self.con = con
        self._whatif = whatif
        self._hubs: pd.DataFrame | None = None

    # ------------------------------------------------------------ lookups ----
    def hubs(self) -> pd.DataFrame:
        if self._hubs is None:
            self._hubs = self.con.execute("SELECT hub_id, name, region FROM hubs").df()
        return self._hubs

    def _q(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    @staticmethod
    def _where(clauses: list[str]) -> str:
        return (" WHERE " + " AND ".join(clauses)) if clauses else ""

    # -------------------------------------------------------------- tools ----
    def get_hub_summary(self) -> dict:
        df = self._q(
            "SELECT hub_id, name, region, assets_at_risk, urgent_jobs, "
            "projected_stockouts, excess_value FROM v_hub_risk ORDER BY urgent_jobs DESC"
        )
        r = _rows(df)
        top = r[0] if r else None
        return {
            "summary": f"{len(r)} hubs; most urgent jobs: {top['name']} ({top['urgent_jobs']})" if top else "no hubs",
            "rows": r,
            "total_urgent_jobs": int(sum(x["urgent_jobs"] for x in r)),
            "hub_count": len(r),
        }

    def get_fleet_risk(self, hub_id: str | None = None, bucket: str | None = None, limit: int = 10) -> dict:
        cl, p = [], []
        if hub_id:
            cl.append("hub_id = ?"); p.append(hub_id)
        if bucket:
            cl.append("bucket = ?"); p.append(bucket)
        counts = self._q(
            "SELECT bucket, COUNT(*) AS assets FROM v_asset_risk" + self._where(cl) + " GROUP BY bucket", p
        )
        df = self._q(
            "SELECT asset_id, asset_label, asset_type, hub_id, criticality, "
            "top_risk_component_type AS component, p10_days, p50_days, p90_days, "
            "p_fail_14d, p_fail_90d, confidence, bucket FROM v_asset_risk"
            + self._where(cl) + " ORDER BY p50_days ASC LIMIT ?", p + [min(int(limit), MAX_ROWS)],
        )
        by = {r["bucket"]: r["assets"] for r in _rows(counts)}
        return {
            "summary": "assets by bucket: " + ", ".join(f"{k} {v}" for k, v in by.items()),
            "counts_by_bucket": by,
            "rows": _rows(df),
        }

    def get_asset_ttm(self, asset_id: str) -> dict:
        df = self._q(
            "SELECT c.component_type, c.p10_days, c.p50_days, c.p90_days, c.p_fail_14d, "
            "c.p_fail_30d, c.p_fail_90d, c.confidence, c.data_quality, c.beyond_horizon, "
            "a.asset_label, a.asset_type, a.hub_id, a.criticality "
            "FROM component_ttm c JOIN assets a ON a.asset_id = c.asset_id "
            "WHERE c.asset_id = ? ORDER BY c.p50_days ASC",
            [asset_id],
        )
        r = _rows(df, 8)
        return {
            "summary": (f"{asset_id}: soonest {r[0]['component_type']} P50 {r[0]['p50_days']} d" if r
                        else f"no asset {asset_id}"),
            "rows": r,
        }

    def get_parts_demand(self, hub_id: str | None = None, part_no: str | None = None,
                         bucket: str | None = None) -> dict:
        cl, p = [], []
        if hub_id:
            cl.append("hub_id = ?"); p.append(hub_id)
        if part_no:
            cl.append("part_no = ?"); p.append(part_no)
        if bucket:
            cl.append("bucket = ?"); p.append(bucket)
        df = self._q(
            "SELECT part_no, description, hub_id, bucket, demand_p10, demand_p50, demand_p90, "
            "criticality, unit_cost FROM parts_demand" + self._where(cl)
            + " ORDER BY demand_p90 DESC", p,
        )
        r = _rows(df)
        return {
            "summary": (f"{len(df)} demand lines; top {r[0]['part_no']} {r[0]['bucket']} "
                        f"P50 {r[0]['demand_p50']} / P90 {r[0]['demand_p90']}" if r else "no matching demand"),
            "rows": r,
            "row_count": len(df),
        }

    def get_stock_position(self, hub_id: str | None = None, part_no: str | None = None,
                           shortfall_only: bool = False) -> dict:
        cl, p = [], []
        if hub_id:
            cl.append("hub_id = ?"); p.append(hub_id)
        if part_no:
            cl.append("part_no = ?"); p.append(part_no)
        if shortfall_only:
            cl.append("shortfall_vs_rop > 0")
        df = self._q(
            "SELECT part_no, hub_id, criticality, on_hand, on_order, reserved, available, "
            "reorder_point, safety_stock, shortfall_vs_rop, shortfall_vs_p90, excess_units, "
            "days_of_cover, lead_time_mean_d, service_level_target FROM stock_positions"
            + self._where(cl) + " ORDER BY shortfall_vs_rop DESC", p,
        )
        r = _rows(df)
        return {
            "summary": (f"{len(df)} stock lines; worst shortfall vs reorder point "
                        f"{r[0]['part_no']} @ {r[0]['hub_id']}: {r[0]['shortfall_vs_rop']}" if r
                        else "no matching stock lines"),
            "rows": r,
            "row_count": len(df),
        }

    def get_recommendations(self, hub_id: str | None = None, part_no: str | None = None,
                            action: str | None = None, limit: int = 10) -> dict:
        cl, p = [], []
        if hub_id:
            cl.append("(to_hub_id = ? OR from_hub_id = ?)"); p += [hub_id, hub_id]
        if part_no:
            cl.append("part_no = ?"); p.append(part_no)
        if action:
            cl.append("action = ?"); p.append(action)
        df = self._q(
            "SELECT rec_id, action, part_no, description, qty, from_hub_id, to_hub_id, "
            "need_in_days, criticality, confidence, why, cost_delta, action_cost, avoided_action "
            "FROM recommendations" + self._where(cl) + " ORDER BY need_in_days ASC LIMIT ?",
            p + [min(int(limit), MAX_ROWS)],
        )
        r = _rows(df)
        return {
            "summary": f"{len(r)} recommendations" + (f"; first {r[0]['rec_id']} {r[0]['action']} {r[0]['qty']} x {r[0]['part_no']}" if r else ""),
            "rows": r,
        }

    def get_supplier(self, part_no: str) -> dict:
        df = self._q(
            "SELECT s.part_no, p.description, s.lead_time_mean_d, s.lead_time_sd_d, s.moq, "
            "s.expedite_cost, s.on_time_rate FROM suppliers s JOIN parts p ON p.part_no = s.part_no "
            "WHERE s.part_no = ?", [part_no])
        r = _rows(df)
        return {
            "summary": (f"{part_no}: lead time {r[0]['lead_time_mean_d']} d, MOQ {r[0]['moq']}, "
                        f"on-time {r[0]['on_time_rate']}" if r else f"no supplier for {part_no}"),
            "rows": r,
        }

    def get_transfer_options(self, to_hub_id: str) -> dict:
        df = self._q(
            "SELECT from_hub_id, to_hub_id, distance_km, transfer_days, same_region "
            "FROM hub_transfers WHERE to_hub_id = ? ORDER BY transfer_days ASC", [to_hub_id])
        r = _rows(df)
        return {"summary": f"{len(r)} source hubs for {to_hub_id}; nearest {r[0]['from_hub_id']} in {r[0]['transfer_days']} d" if r else "no transfer lanes", "rows": r}

    def get_evidence_summary(self) -> dict:
        out: dict[str, Any] = {}
        try:
            out["ims_bearings"] = _rows(self._q(
                "SELECT test, failure_mode, life_days, lead_time_days, rank_corr_hi_vs_rul "
                "FROM evidence_ims_summary ORDER BY test"))
        except Exception:
            pass
        try:
            out["ashrae_baseline_sd_pct"] = _clean(self._q(
                "SELECT baseline_resid_sd_pct AS v FROM evidence_ashrae LIMIT 1")["v"].iloc[0])
            out["ashrae_runs"] = int(self._q("SELECT COUNT(DISTINCT file) AS n FROM evidence_ashrae")["n"].iloc[0])
        except Exception:
            pass
        return {"summary": "real-data validation: " + ", ".join(out) if out else "no evidence loaded", "rows": [out]}

    def run_whatif(self, asset_id: str, d_ambient: float = 0.0, d_load: float = 0.0) -> dict:
        if self._whatif is None:
            return {"summary": "what-if engine unavailable", "rows": []}
        res = self._whatif(asset_id=asset_id, d_ambient=float(d_ambient), d_load=float(d_load))
        before = {c["component_id"]: c for c in res.get("components_before", [])}
        comps = []
        for c in res.get("components_after", []):
            b0 = before.get(c["component_id"], {})
            comps.append({
                "component_type": c["component_type"],
                "p50_days_before": b0.get("p50_days"), "p50_days_after": c["p50_days"],
                "p_fail_14d_before": b0.get("p_fail_14d"), "p_fail_14d_after": c["p_fail_14d"],
                "bucket_before": b0.get("bucket"), "bucket_after": c["bucket"],
            })
        comps.sort(key=lambda x: x["p50_days_after"])
        slim = {
            "scenario": res.get("scenario_label"),
            "asset_label": res.get("asset_label"),
            "bucket_counts_before": res.get("bucket_counts_before"),
            "bucket_counts_after": res.get("bucket_counts_after"),
            "bucket_migration": res.get("bucket_migration"),
            "components": comps[:5],
            "parts_delta": res.get("parts_delta", [])[:5],
            "held_fixed": res.get("assumptions", [])[:3],
        }
        return {"summary": f"what-if {asset_id}: {res.get('scenario_label')}", "rows": [_clean_deep(slim)]}


def _clean_deep(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _clean_deep(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_deep(v) for v in o]
    return _clean(o)


# --------------------------------------------------------------------------
# Registry: JSON-schema tool definitions (Anthropic format). The `description`
# says WHEN to call the tool, not just what it does - models reach for tools
# conservatively and trigger conditions in the description lift the call rate.
# --------------------------------------------------------------------------

_HUB = {"type": "string", "description": "Hub id such as HUB-PNQ (Pune). Omit for the whole network."}
_PART = {"type": "string", "description": "Part number such as CNT-3P-40A or CMP-M."}
_BUCKET = {"type": "string", "enum": ["urgent", "planned", "forecast"],
           "description": "urgent = 0-14 d, planned = 15-90 d, forecast = beyond 90 d"}


def _t(name: str, desc: str, props: dict, req: list[str] | None = None) -> dict:
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props, "required": req or []}}


TOOL_SPECS: dict[str, dict] = {s["name"]: s for s in [
    _t("get_hub_summary", "Call first for any network-wide question: urgent jobs, assets at risk, projected stockouts and excess stock per hub.", {}),
    _t("get_fleet_risk", "Call when asked which assets or how many are at risk, urgent, or due soon, for a hub or the network. Returns bucket counts and the soonest-to-fail assets with P10/P50/P90 days.",
       {"hub_id": _HUB, "bucket": _BUCKET, "limit": {"type": "integer"}}),
    _t("get_asset_ttm", "Call when a specific asset id (e.g. CH-117, RTU-118) is mentioned, or to explain why an asset is flagged. Returns each component's time-to-maintenance band and failure probabilities.",
       {"asset_id": {"type": "string"}}, ["asset_id"]),
    _t("get_parts_demand", "Call when asked how many of a part a hub will need, or what is driving demand. Monte Carlo P10/P50/P90 per part, hub and horizon bucket.",
       {"hub_id": _HUB, "part_no": _PART, "bucket": _BUCKET}),
    _t("get_stock_position", "Call when asked about stock on hand, shortages, cover, or reorder points. Returns on-hand, on-order, reorder point, safety stock and shortfall.",
       {"hub_id": _HUB, "part_no": _PART, "shortfall_only": {"type": "boolean"}}),
    _t("get_recommendations", "Call when asked what to do about a shortage: the transfer / purchase / expedite / defer cards with quantity, need-by, cost and reason.",
       {"hub_id": _HUB, "part_no": _PART, "action": {"type": "string", "enum": ["transfer", "purchase", "expedite", "defer"]}, "limit": {"type": "integer"}}),
    _t("get_supplier", "Call when lead time, minimum order quantity, expedite cost or supplier reliability matters to a part decision.",
       {"part_no": _PART}, ["part_no"]),
    _t("get_transfer_options", "Call when considering moving stock between hubs: source hubs for a destination, with distance and transfer days.",
       {"to_hub_id": _HUB}, ["to_hub_id"]),
    _t("get_evidence_summary", "Call when asked how much the predictions can be trusted or what real-world data validates the method (NASA IMS bearings, ASHRAE chiller faults).", {}),
    _t("run_whatif", "Call when asked what happens under a hotter ambient temperature or a higher load for one asset. Re-scores that asset's components under the scenario.",
       {"asset_id": {"type": "string"}, "d_ambient": {"type": "number", "description": "deg C change"}, "d_load": {"type": "number"}}, ["asset_id"]),
]}


def call_tool(box: ToolBox, name: str, args: dict) -> tuple[dict, float, bool]:
    """Run a tool; returns (result, duration_ms, is_error). Never raises."""
    t0 = time.perf_counter()
    try:
        fn = getattr(box, name)
        res = fn(**{k: v for k, v in args.items() if v is not None})
        err = False
    except Exception as exc:  # a bad argument from the model must not kill the turn
        res, err = {"summary": f"tool error: {exc}", "rows": []}, True
    res.setdefault("rows", [])
    return res, (time.perf_counter() - t0) * 1000, err
