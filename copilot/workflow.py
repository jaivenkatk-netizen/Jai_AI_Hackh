"""Multi-agent copilot workflow.

    question
       |
    Orchestrator ---- decides which specialists are needed (a plan)
       |
       +-- Risk Analyst      fleet risk, asset time-to-maintenance, what-if
       +-- Parts Planner     parts demand and stock position
       +-- Logistics Agent   recommendations, suppliers, transfer lanes
       +-- Validation Agent  how far the predictions can be trusted
       |
    Synthesizer ---- writes one answer from the specialists' findings only
       |
    Auditor ---- checks every number in the answer against a tool result

The rule that makes it credible is the same one the dashboard follows: the
copilot answers only from tool results, and every tool call is streamed to the
UI beside the answer. Nothing here writes to the database or approves an
action - it explains and recommends; a planner still clicks Approve.

Two engines share this flow:

  * "claude"  - each agent is a Claude call with its own system prompt and its
                own tool subset. Used when ANTHROPIC_API_KEY is set.
  * "offline" - a deterministic router and templated synthesiser that call the
                same tools. Used with no key, so the demo never depends on a
                network call. It is rule-based, not a language model, and the
                UI says so.

Both stream the same events, so the chat panel does not care which is running.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

from copilot.tools import TOOL_SPECS, ToolBox, call_tool

MODEL = os.environ.get("COPILOT_MODEL", "claude-opus-5-5")
MAX_TOOL_TURNS = 5


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Agent:
    key: str
    name: str
    role: str
    tools: tuple[str, ...]
    system: str


AGENTS: dict[str, Agent] = {a.key: a for a in [
    Agent(
        "risk_analyst", "Risk Analyst", "Which assets are failing, and when",
        ("get_fleet_risk", "get_asset_ttm", "run_whatif", "get_hub_summary"),
        "You are the Risk Analyst for an HVAC maintenance-to-inventory system. You report which "
        "assets or components are likely to need maintenance and when, using time-to-maintenance "
        "bands (P10/P50/P90 days) and failure probabilities. Always call a tool before stating any "
        "number. State the band, not just the median, and mention confidence when it is not high. "
        "Reply with 2-4 short factual sentences. Never invent an asset id.",
    ),
    Agent(
        "parts_planner", "Parts Planner", "What parts that implies, and the stock position",
        ("get_parts_demand", "get_stock_position"),
        "You are the Parts Planner. You translate predicted failures into part quantities and compare "
        "them with stock. Always call a tool before stating a number. Give demand as P50 with the P90 "
        "upper case, and compare against on-hand, on-order and the reorder point. Reply with 2-4 short "
        "factual sentences.",
    ),
    Agent(
        "logistics", "Logistics Agent", "What to do about it: transfer, buy or expedite",
        ("get_recommendations", "get_supplier", "get_transfer_options"),
        "You are the Logistics Agent. You report the recommended stock action (transfer, purchase, "
        "expedite, defer) for a shortage and whether supplier lead time fits the need date. Always call "
        "a tool before stating a number. Recommendations are proposals, not approved actions - say so. "
        "Reply with 2-4 short factual sentences.",
    ),
    Agent(
        "validator", "Validation Agent", "How far the predictions can be trusted",
        ("get_evidence_summary",),
        "You are the Validation Agent. You report how the method performs on real public data (NASA IMS "
        "bearing run-to-failure tests and ASHRAE chiller fault tests). Be candid about weak results. "
        "Always call a tool before stating a number. Reply with 2-4 short factual sentences.",
    ),
]}

ORCHESTRATOR_SYSTEM = (
    "You are the Orchestrator of a multi-agent HVAC maintenance-to-inventory copilot. Decide which "
    "specialist agents are needed for the user's question. Specialists: "
    + "; ".join(f"{a.key} ({a.role})" for a in AGENTS.values())
    + ". Reply with JSON only: {\"agents\": [\"risk_analyst\", ...], \"reason\": \"one sentence\"}. "
    "Pick the fewest agents that can answer. Use screen context (hub, asset) when the question is vague."
)
SYNTH_SYSTEM = (
    "You are the Synthesizer. Write the final answer for a maintenance planner using ONLY the "
    "specialists' findings you are given. Lead with the direct answer, then the supporting numbers, "
    "then one next step. Do not add any number or fact that is not in the findings. Plain text, under "
    "120 words, no markdown headings."
)


# --------------------------------------------------------------------------
# Entity extraction (offline router, and context for the LLM planner)
# --------------------------------------------------------------------------


@dataclass
class Entities:
    hub_id: str | None = None
    part_no: str | None = None
    asset_id: str | None = None
    unknown_asset: str | None = None
    bucket: str | None = None
    d_ambient: float | None = None
    wants: set[str] = field(default_factory=set)


def extract_entities(box: ToolBox, message: str, context: dict[str, Any]) -> Entities:
    text = message.lower()
    e = Entities()
    for _, h in box.hubs().iterrows():
        if h["name"].lower() in text or h["hub_id"].lower() in text:
            e.hub_id = h["hub_id"]
            break
    parts = box.con.execute("SELECT part_no, lower(description) AS d FROM parts").fetchall()
    for pn, d in parts:
        if pn.lower() in text:
            e.part_no = pn
            break
    if not e.part_no:
        # "contactor" -> the part whose description contains it; skip ambiguous matches
        for word in re.findall(r"[a-z]{4,}", text):
            word = word[:-1] if word.endswith("s") and len(word) > 4 else word
            hits = [pn for pn, d in parts if word in d]
            if len(hits) == 1:
                e.part_no = hits[0]
                break
    m = re.search(r"\b([A-Za-z]{2,5}-\d{2,4})\b", message)
    if m and not box.con.execute("SELECT 1 FROM assets WHERE upper(asset_label) = ? OR asset_id = ?",
                                 [m.group(1).upper(), m.group(1).upper()]).fetchone():
        e.unknown_asset = m.group(1).upper()
    if m and not e.unknown_asset:
        e.asset_id = m.group(1).upper()
        row = box.con.execute("SELECT asset_id FROM assets WHERE upper(asset_label) = ?",
                              [e.asset_id]).fetchone()
        if row:
            e.asset_id = row[0]
    for b in ("urgent", "planned", "forecast"):
        if b in text:
            e.bucket = b
    if not e.bucket and re.search(r"next week|this week|soon|immediate|right now|asap|tomorrow", text):
        e.bucket = "urgent"
    if re.search(r"heat ?wave|hotter|ambient|degrees|°c|what if|what-if", text):
        e.wants.add("whatif")
        mm = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:°|deg)", text)
        e.d_ambient = float(mm.group(1)) if mm else 6.0
    # fall back to the screen the user is looking at
    e.hub_id = e.hub_id or context.get("hub_id")
    e.asset_id = e.asset_id or context.get("asset_id")
    kw = {
        "risk": r"risk|fail|urgent|due|which assets|how many assets|break|ttm|maintenance|why",
        "parts": r"part|need|demand|units|stock|shortage|short of|inventory|cover|reorder|how many",
        "logistics": r"recommend|transfer|buy|purchase|order|expedite|supplier|lead time|should we|what to do|action",
        "trust": r"trust|valid|evidence|accura|reliab|confiden|real data|proof|how good",
    }
    for k, pat in kw.items():
        if re.search(pat, text):
            e.wants.add(k)
    return e


def plan_offline(e: Entities) -> tuple[list[str], str]:
    agents: list[str] = []
    if e.asset_id or e.bucket or "risk" in e.wants or "whatif" in e.wants:
        agents.append("risk_analyst")
    if e.part_no or "parts" in e.wants:
        agents.append("parts_planner")
    if "logistics" in e.wants or (e.part_no and "parts" in e.wants):
        agents.append("logistics")
    if "trust" in e.wants:
        agents.append("validator")
    if not agents:
        agents = ["risk_analyst", "parts_planner"]
        reason = "No specific target found, so a network overview from risk and parts."
    else:
        reason = "Routed on the entities and intent found in the question."
    return list(dict.fromkeys(agents)), reason


# --------------------------------------------------------------------------
# Offline specialists: playbooks of tool calls + findings written from rows
# --------------------------------------------------------------------------


def _f(v: Any, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        s = f"{v:.{nd}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s
    return str(v)


def _describe(tool: str, args: dict, res: dict) -> list[str]:
    rows = res.get("rows", [])
    if tool == "get_hub_summary" and rows:
        top = rows[0]
        return [f"{top['name']} has the most urgent jobs ({top['urgent_jobs']}); network-wide "
                f"{res['total_urgent_jobs']} urgent jobs across {res['hub_count']} hubs."]
    if tool == "get_fleet_risk":
        c = res.get("counts_by_bucket", {})
        out = []
        if c:
            where = f"in {args['hub_id']}" if args.get("hub_id") else "network-wide"
            out.append(f"{where}: " + ", ".join(f"{v} assets {k}" for k, v in c.items()) + ".")
        for r in rows[:3]:
            out.append(f"{r['asset_label']} ({r['component']}) P50 {_f(r['p50_days'])} d "
                       f"[P10 {_f(r['p10_days'])}, P90 {_f(r['p90_days'])}], "
                       f"P(fail 14 d) {_f(r['p_fail_14d'], 2)}, confidence {r['confidence']}.")
        return out
    if tool == "get_asset_ttm":
        if not rows:
            return [f"No asset {args.get('asset_id')} found."]
        h = rows[0]
        out = [f"{h['asset_label']} ({h['asset_type']}, hub {h['hub_id']}, criticality {h['criticality']})."]
        for r in rows[:3]:
            out.append(f"{r['component_type']}: P50 {_f(r['p50_days'])} d "
                       f"[P10 {_f(r['p10_days'])}, P90 {_f(r['p90_days'])}], "
                       f"P(fail 14 d) {_f(r['p_fail_14d'], 2)}, confidence {r['confidence']}.")
        return out
    if tool == "run_whatif" and rows:
        w = rows[0]
        out = [f"What-if {w.get('scenario')}: "]
        for c in (w.get("components") or [])[:3]:
            out.append(f"{c['component_type']} P50 {_f(c['p50_days_before'])} d -> {_f(c['p50_days_after'])} d, "
                       f"P(fail 14 d) {_f(c['p_fail_14d_before'], 2)} -> {_f(c['p_fail_14d_after'], 2)}.")
        mig = w.get("bucket_migration") or {}
        if mig:
            out.append("Bucket moves: " + ", ".join(f"{k} x{v}" for k, v in mig.items()) + ".")
        for d in (w.get("parts_delta") or [])[:2]:
            out.append(f"{d['part_no']} P90 demand {_f(d['p90_before'])} -> {_f(d['p90_after'])}.")
        return [" ".join(out)]
    if tool == "get_parts_demand":
        if not rows:
            return ["No parts demand matches that filter."]
        return [f"{r['part_no']} ({r['description']}) at {r['hub_id']}, {r['bucket']}: "
                f"P50 {_f(r['demand_p50'])} units, P90 {_f(r['demand_p90'])}." for r in rows[:3]]
    if tool == "get_stock_position":
        if not rows:
            return ["No stock line matches that filter."]
        out = []
        for r in rows[:3]:
            gap = (f"short {_f(r['shortfall_vs_rop'])} vs reorder point"
                   if (r["shortfall_vs_rop"] or 0) > 0 else "at or above reorder point")
            out.append(f"{r['part_no']} at {r['hub_id']}: on hand {_f(r['on_hand'])}, "
                       f"on order {_f(r['on_order'])}, reorder point {_f(r['reorder_point'])}, {gap}.")
        return out
    if tool == "get_recommendations":
        if not rows:
            return ["No recommendation card matches that filter."]
        out = []
        for r in rows[:3]:
            move = (f"from {r['from_hub_id']} " if r.get("from_hub_id") else "") + (
                f"to {r['to_hub_id']}" if r.get("to_hub_id") else "")
            out.append(f"{r['rec_id']}: {r['action']} {_f(r['qty'])} x {r['part_no']} {move}, "
                       f"needed in {_f(r['need_in_days'])} d, confidence {r['confidence']}. "
                       "Proposed, not yet approved.")
        return out
    if tool == "get_supplier" and rows:
        r = rows[0]
        return [f"{r['part_no']} supplier lead time {_f(r['lead_time_mean_d'])} d "
                f"(sd {_f(r['lead_time_sd_d'])}), MOQ {_f(r['moq'])}, on-time rate {_f(r['on_time_rate'], 2)}."]
    if tool == "get_transfer_options" and rows:
        r = rows[0]
        return [f"Nearest source hub for {r['to_hub_id']} is {r['from_hub_id']}, "
                f"{_f(r['distance_km'], 0)} km, {_f(r['transfer_days'])} d."]
    if tool == "get_evidence_summary" and rows:
        ev = rows[0]
        out = []
        for t in ev.get("ims_bearings", []):
            out.append(f"NASA IMS test {t['test']}: alarm {_f(t['lead_time_days'])} d before failure "
                       f"in a {_f(t['life_days'])} d run, rank correlation {_f(t['rank_corr_hi_vs_rul'], 2)}.")
        if "ashrae_runs" in ev:
            out.append(f"ASHRAE: {ev['ashrae_runs']} chiller fault runs; fault-free efficiency scatter "
                       f"{_f(ev.get('ashrae_baseline_sd_pct'), 1)} percent.")
        return out
    return [res.get("summary", "")]


def offline_playbook(agent_key: str, e: Entities) -> list[tuple[str, dict]]:
    if agent_key == "risk_analyst":
        if e.asset_id:
            calls = [("get_asset_ttm", {"asset_id": e.asset_id})]
            if "whatif" in e.wants:
                calls.append(("run_whatif", {"asset_id": e.asset_id, "d_ambient": e.d_ambient or 6.0}))
            return calls
        if not (e.hub_id or e.bucket or "risk" in e.wants):
            return [("get_hub_summary", {})]
        return [("get_fleet_risk", {"hub_id": e.hub_id, "bucket": e.bucket or "urgent", "limit": 5})]
    if agent_key == "parts_planner":
        # Default to the urgent horizon: it is the one a planner can still act on.
        calls = [("get_parts_demand", {"hub_id": e.hub_id, "part_no": e.part_no, "bucket": e.bucket or "urgent"})]
        calls.append(("get_stock_position", {"hub_id": e.hub_id, "part_no": e.part_no,
                                              "shortfall_only": not e.part_no}))
        return calls
    if agent_key == "logistics":
        calls = [("get_recommendations", {"hub_id": e.hub_id, "part_no": e.part_no, "limit": 5})]
        if e.part_no:
            calls.append(("get_supplier", {"part_no": e.part_no}))
        if e.hub_id:
            calls.append(("get_transfer_options", {"to_hub_id": e.hub_id}))
        return calls
    if agent_key == "validator":
        return [("get_evidence_summary", {})]
    return []


# --------------------------------------------------------------------------
# Auditor: every number in the answer must come from a tool result
# --------------------------------------------------------------------------

_ID = re.compile(r"\b[A-Za-z]{2,}[-_][A-Za-z0-9-]*\d[A-Za-z0-9-]*\b|\b[A-Z]{2,4}-\d+\b")
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _pool(obj: Any, out: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
        if 0 <= obj <= 1:
            out.add(float(obj) * 100)
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            out.add(float(m))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            for m in re.findall(r"\d+", str(k)):  # horizon in a field name, e.g. p_fail_14d
                out.add(float(m))
            _pool(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _pool(v, out)


def audit(answer: str, results: list[dict], question: str, context: dict) -> dict:
    pool: set[float] = set()
    for r in results:
        _pool(r, out=pool)
        pool.add(float(len(r.get("rows", []))))
    allowed: set[float] = set()
    _pool(question, allowed)
    _pool(context, allowed)
    checked, ungrounded = 0, []
    for tok in _NUM.findall(_ID.sub(" ", answer)):
        x = float(tok)
        checked += 1
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        if x in allowed or any(abs(x - p) <= tol for p in pool):
            continue
        ungrounded.append(tok)
    return {"checked": checked, "ungrounded": ungrounded, "grounded": not ungrounded and bool(results)}


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------


def engine_status() -> dict:
    live = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return {
        "engine": "claude" if live else "offline",
        "model": MODEL if live else None,
        "note": ("Each agent is a Claude call with its own tools." if live else
                 "No ANTHROPIC_API_KEY set: a deterministic router and templated writer call the same "
                 "tools. It is rule-based, not a language model."),
        "agents": [{"key": a.key, "name": a.name, "role": a.role, "tools": list(a.tools)}
                   for a in AGENTS.values()],
    }


def _claude_text(client, system: str, user: str, max_tokens: int = 600) -> str:
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                               messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in r.content if b.type == "text").strip()


def _claude_specialist(client, agent: Agent, question: str, ctx: dict, box: ToolBox,
                       emit: list[dict], results: list[dict]) -> str:
    tools = [TOOL_SPECS[t] for t in agent.tools]
    msgs: list[dict] = [{"role": "user", "content": f"Question: {question}\nScreen context: {json.dumps(ctx)}"}]
    for _ in range(MAX_TOOL_TURNS):
        r = client.messages.create(model=MODEL, max_tokens=900, system=agent.system,
                                   tools=tools, messages=msgs)
        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if b.type == "text").strip()
        msgs.append({"role": "assistant", "content": r.content})
        outs = []
        for b in r.content:
            if b.type != "tool_use":
                continue
            emit.append({"type": "tool_call", "agent": agent.key, "tool": b.name, "arguments": b.input})
            res, ms, err = call_tool(box, b.name, dict(b.input))
            results.append(res)
            emit.append({"type": "tool_result", "agent": agent.key, "tool": b.name,
                         "summary": res.get("summary", ""), "row_count": len(res["rows"]),
                         "duration_ms": round(ms, 1), "is_error": err})
            outs.append({"type": "tool_result", "tool_use_id": b.id,
                         "content": json.dumps(res["rows"])[:6000], "is_error": err})
        msgs.append({"role": "user", "content": outs})
    return "(stopped after the tool-call limit)"


def run_workflow(box: ToolBox, message: str, context: dict[str, Any] | None = None,
                 force_offline: bool = False) -> Iterator[dict]:
    """Yield workflow events. Never raises: failures become an `error` event."""
    context = context or {}
    live = bool(os.environ.get("ANTHROPIC_API_KEY")) and not force_offline
    client = None
    engine = "offline"
    if live:
        try:
            import anthropic
            client = anthropic.Anthropic()
            engine = "claude"
        except Exception as exc:
            yield {"type": "error", "agent": "orchestrator", "text": f"Claude unavailable ({exc}); using offline engine."}
    yield {"type": "start", "engine": engine, "model": MODEL if client else None}

    results: list[dict] = []
    findings: list[tuple[str, str]] = []
    ents = extract_entities(box, message, context)

    # ---- Orchestrator
    try:
        if client:
            raw = _claude_text(client, ORCHESTRATOR_SYSTEM,
                               f"Question: {message}\nScreen context: {json.dumps(context)}\n"
                               f"Entities found: hub={ents.hub_id} part={ents.part_no} asset={ents.asset_id}",
                               200)
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            plan = [a for a in data.get("agents", []) if a in AGENTS] or ["risk_analyst", "parts_planner"]
            reason = str(data.get("reason", ""))
        else:
            plan, reason = plan_offline(ents)
    except Exception as exc:
        plan, reason = plan_offline(ents)
        yield {"type": "error", "agent": "orchestrator", "text": f"Planner fell back to rules ({exc})."}
    yield {"type": "plan", "agent": "orchestrator", "agents": plan, "reason": reason,
           "entities": {"hub_id": ents.hub_id, "part_no": ents.part_no, "asset_id": ents.asset_id,
                        "bucket": ents.bucket}}

    if ents.unknown_asset and not ents.asset_id:
        # Do not answer a different question than the one asked.
        msg = (f"There is no asset named {ents.unknown_asset} in the fleet. Asset ids look like "
               "CH-406, RTU-199 or VRF-403; name one, or open an asset from the Command Center.")
        yield {"type": "agent_start", "agent": "risk_analyst", "name": "Risk Analyst", "role": AGENTS["risk_analyst"].role}
        yield {"type": "finding", "agent": "risk_analyst", "text": msg}
        yield {"type": "verify", "agent": "auditor", "checked": 0, "ungrounded": [], "grounded": True}
        yield {"type": "text", "agent": "synthesizer", "text": msg}
        yield {"type": "done", "grounded": True, "engine": engine}
        return

    # ---- Specialists
    for key in plan:
        agent = AGENTS[key]
        yield {"type": "agent_start", "agent": key, "name": agent.name, "role": agent.role}
        try:
            if client:
                buf: list[dict] = []
                text = _claude_specialist(client, agent, message, context, box, buf, results)
                yield from buf
                findings.append((key, text))
                yield {"type": "finding", "agent": key, "text": text}
            else:
                lines: list[str] = []
                for tool, args in offline_playbook(key, ents):
                    args = {k: v for k, v in args.items() if v is not None}
                    yield {"type": "tool_call", "agent": key, "tool": tool, "arguments": args}
                    res, ms, err = call_tool(box, tool, args)
                    results.append(res)
                    yield {"type": "tool_result", "agent": key, "tool": tool,
                           "summary": res.get("summary", ""), "row_count": len(res["rows"]),
                           "duration_ms": round(ms, 1), "is_error": err}
                    if not err:
                        lines += _describe(tool, args, res)
                text = " ".join(l for l in lines if l)
                findings.append((key, text))
                yield {"type": "finding", "agent": key, "text": text}
        except Exception as exc:
            yield {"type": "error", "agent": key, "text": f"{agent.name} failed: {exc}"}

    # ---- Synthesizer
    yield {"type": "agent_start", "agent": "synthesizer", "name": "Synthesizer",
           "role": "Writes one answer from the findings"}
    if not any(t for _, t in findings):
        answer = "I could not find data to answer that. Try naming a hub, part number or asset id."
    elif client:
        try:
            body = "\n".join(f"[{AGENTS[k].name}] {t}" for k, t in findings)
            answer = _claude_text(client, SYNTH_SYSTEM, f"Question: {message}\n\nFindings:\n{body}")
        except Exception as exc:
            yield {"type": "error", "agent": "synthesizer", "text": f"Synthesizer fell back ({exc})."}
            answer = " ".join(t for _, t in findings if t)
    else:
        answer = " ".join(t for _, t in findings if t)

    # ---- Auditor
    yield {"type": "agent_start", "agent": "auditor", "name": "Auditor",
           "role": "Checks every number against a tool result"}
    verdict = audit(answer, results, message, context)
    if client and verdict["ungrounded"]:
        try:
            body = "\n".join(f"[{AGENTS[k].name}] {t}" for k, t in findings)
            answer = _claude_text(
                client, SYNTH_SYSTEM,
                f"Question: {message}\n\nFindings:\n{body}\n\nYour draft used numbers that are not in the "
                f"findings: {verdict['ungrounded']}. Rewrite it using only numbers from the findings.\n\nDraft: {answer}")
            verdict = audit(answer, results, message, context) | {"revised": True}
        except Exception:
            pass
    yield {"type": "verify", "agent": "auditor", **verdict}
    yield {"type": "text", "agent": "synthesizer", "text": answer}
    yield {"type": "done", "grounded": verdict["grounded"], "engine": engine}
