// Screen 6 - Copilot. A chat interface over a multi-agent workflow: an
// orchestrator routes the question to specialist agents, each calls read-only
// tools, a synthesizer writes one answer and an auditor checks every number
// against a tool result. The workflow is rendered beside the answer so nothing
// the copilot says is untraceable.

import { useEffect, useRef, useState } from "react";

import { Card, Pill } from "../components/ui";

interface WfEvent {
  type: string;
  agent?: string;
  name?: string;
  role?: string;
  text?: string;
  agents?: string[];
  reason?: string;
  entities?: Record<string, string | null>;
  tool?: string;
  arguments?: Record<string, unknown>;
  summary?: string;
  row_count?: number;
  duration_ms?: number;
  is_error?: boolean;
  checked?: number;
  ungrounded?: string[];
  grounded?: boolean;
  engine?: string;
  model?: string | null;
  revised?: boolean;
}

interface Turn {
  id: number;
  question: string;
  events: WfEvent[];
  answer: string | null;
  done: boolean;
  grounded: boolean | null;
  engine: string | null;
}

interface Status {
  engine: string;
  model: string | null;
  note: string;
  agents: { key: string; name: string; role: string; tools: string[] }[];
}

const EXAMPLES = [
  "Why do we need contactors in Pune next week?",
  "How many assets are urgent in Chennai?",
  "What should we do about shortages in Mumbai?",
  "How much can I trust these predictions?",
];

const AGENT_NAME: Record<string, string> = {
  orchestrator: "Orchestrator",
  risk_analyst: "Risk Analyst",
  parts_planner: "Parts Planner",
  logistics: "Logistics Agent",
  validator: "Validation Agent",
  synthesizer: "Synthesizer",
  auditor: "Auditor",
};

async function streamCopilot(
  message: string,
  context: Record<string, unknown>,
  onEvent: (e: WfEvent) => void,
): Promise<void> {
  const r = await fetch("/api/copilot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, context, history: [] }),
  });
  if (!r.ok || !r.body) throw new Error(`${r.status} ${r.statusText}`);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, i);
      buf = buf.slice(i + 2);
      if (frame.startsWith("data: ")) onEvent(JSON.parse(frame.slice(6)) as WfEvent);
    }
  }
}

function Workflow({ turn }: { turn: Turn }) {
  // Group events by agent, in the order each agent first appears.
  const order: string[] = [];
  for (const e of turn.events)
    if (e.agent && e.type !== "start" && !order.includes(e.agent)) order.push(e.agent);
  const plan = turn.events.find((e) => e.type === "plan");
  return (
    <ol className="flex flex-col gap-2 text-[12px]">
      {order.map((key) => {
        const evs = turn.events.filter((e) => e.agent === key);
        const verify = evs.find((e) => e.type === "verify");
        return (
          <li key={key} className="rounded p-2" style={{ background: "var(--plane)" }}>
            <div className="flex items-baseline justify-between gap-2">
              <span className="font-semibold">{AGENT_NAME[key] ?? key}</span>
              {key === "orchestrator" && plan && (
                <span style={{ color: "var(--muted)" }}>
                  routes to {(plan.agents ?? []).map((a) => AGENT_NAME[a] ?? a).join(", ")}
                </span>
              )}
            </div>
            {key === "orchestrator" && plan?.reason && (
              <div style={{ color: "var(--ink-2)" }}>{plan.reason}</div>
            )}
            {evs
              .filter((e) => e.type === "tool_result")
              .map((e, i) => (
                <div key={i} className="mt-1 flex flex-wrap items-baseline gap-x-2">
                  <code className="num">{e.tool}</code>
                  <span style={{ color: e.is_error ? "var(--urgent)" : "var(--ink-2)" }}>
                    {e.summary}
                  </span>
                  <span style={{ color: "var(--muted)" }}>
                    {e.row_count} rows, {e.duration_ms} ms
                  </span>
                </div>
              ))}
            {evs
              .filter((e) => e.type === "finding" && e.text)
              .map((e, i) => (
                <div key={i} className="mt-1" style={{ color: "var(--ink-2)" }}>
                  {e.text}
                </div>
              ))}
            {evs
              .filter((e) => e.type === "error")
              .map((e, i) => (
                <div key={i} className="mt-1" style={{ color: "var(--urgent)" }}>
                  {e.text}
                </div>
              ))}
            {verify && (
              <div className="mt-1" style={{ color: verify.grounded ? "var(--ink-2)" : "var(--urgent)" }}>
                {verify.grounded
                  ? `All ${verify.checked} numbers in the answer match a tool result.`
                  : `Not grounded: ${verify.ungrounded?.join(", ") || "no tool results"} not found in any tool result.`}
                {verify.revised ? " (answer was revised once)" : ""}
              </div>
            )}
          </li>
        );
      })}
      {!turn.done && <li style={{ color: "var(--muted)" }}>Working...</li>}
    </ol>
  );
}

export default function Copilot({ hub, assetId }: { hub: string | null; assetId: string | null }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  useEffect(() => {
    fetch("/api/copilot/status")
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const ask = async (q: string) => {
    const question = q.trim();
    if (!question || busy) return;
    const id = nextId.current++;
    setInput("");
    setBusy(true);
    setTurns((t) => [...t, { id, question, events: [], answer: null, done: false, grounded: null, engine: null }]);
    const patch = (fn: (t: Turn) => Turn) =>
      setTurns((all) => all.map((t) => (t.id === id ? fn(t) : t)));
    try {
      await streamCopilot(
        question,
        { ...(hub ? { hub_id: hub } : {}), ...(assetId ? { asset_id: assetId } : {}) },
        (e) =>
          patch((t) => ({
            ...t,
            events: e.type === "text" || e.type === "done" || e.type === "start" ? t.events : [...t.events, e],
            answer: e.type === "text" ? (e.text ?? null) : t.answer,
            grounded: e.type === "done" ? (e.grounded ?? null) : t.grounded,
            engine: e.type === "start" ? (e.engine ?? null) : t.engine,
            done: t.done || e.type === "done",
          })),
      );
    } catch (err) {
      patch((t) => ({
        ...t,
        events: [...t.events, { type: "error", agent: "orchestrator", text: String(err) }],
        done: true,
      }));
    } finally {
      patch((t) => ({ ...t, done: true }));
      setBusy(false);
    }
  };

  const live = status?.engine === "claude";
  return (
    <div className="mx-auto flex max-w-[1000px] flex-col gap-3">
      <Card
        title="Copilot"
        subtitle="Ask in plain language. A team of agents answers from the same data as the dashboard, and shows its work."
        right={
          status && (
            <Pill title={status.note}>{live ? `Claude (${status.model})` : "Offline rules engine"}</Pill>
          )
        }
      >
        {status && (
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px]" style={{ color: "var(--ink-2)" }}>
            {status.agents.map((a) => (
              <span key={a.key} title={a.tools.join(", ")}>
                <b>{a.name}</b> - {a.role}
              </span>
            ))}
          </div>
        )}
        {status && !live && (
          <p className="mt-2 text-[11px]" style={{ color: "var(--muted)" }}>
            {status.note} Set ANTHROPIC_API_KEY and restart the API to use Claude for each agent.
          </p>
        )}
        {(hub || assetId) && (
          <p className="mt-2 text-[11px]" style={{ color: "var(--muted)" }}>
            Context: {[hub && `hub ${hub}`, assetId && `asset ${assetId}`].filter(Boolean).join(", ")} (used when the question does not name one).
          </p>
        )}
      </Card>

      <div className="flex flex-col gap-3">
        {turns.length === 0 && (
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => ask(q)}
                className="rounded px-3 py-1.5 text-left text-[12px]"
                style={{ background: "var(--plane)", border: "1px solid var(--line)" }}
              >
                {q}
              </button>
            ))}
          </div>
        )}
        {turns.map((t) => (
          <div key={t.id} className="flex flex-col gap-2">
            <div
              className="self-end rounded px-3 py-2 text-[13px]"
              style={{ background: "var(--plane)", maxWidth: "85%" }}
            >
              {t.question}
            </div>
            <Card bodyClass="p-3">
              {t.answer ? (
                <p className="text-[13px] leading-relaxed">{t.answer}</p>
              ) : (
                <p className="text-[12px]" style={{ color: "var(--muted)" }}>
                  {t.done ? "No answer produced." : "The agents are working..."}
                </p>
              )}
              {t.done && t.grounded === false && (
                <p className="mt-2 text-[11px]" style={{ color: "var(--urgent)" }}>
                  The auditor could not tie every number to a tool result. Treat this answer with care.
                </p>
              )}
              <details className="mt-3" open={t.id === turns[turns.length - 1].id}>
                <summary className="cursor-pointer text-[12px]" style={{ color: "var(--ink-2)" }}>
                  How the agents got this
                </summary>
                <div className="mt-2">
                  <Workflow turn={t} />
                </div>
              </details>
            </Card>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
        className="sticky bottom-3 flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about risk, parts, stock or what to do..."
          maxLength={1000}
          className="flex-1 rounded px-3 py-2 text-[13px]"
          style={{ background: "var(--surface)", border: "1px solid var(--line)" }}
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded px-4 py-2 text-[13px] font-semibold"
          style={{ background: "var(--ink)", color: "var(--surface)", opacity: busy || !input.trim() ? 0.5 : 1 }}
        >
          {busy ? "Working" : "Ask"}
        </button>
      </form>
    </div>
  );
}
