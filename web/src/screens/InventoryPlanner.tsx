// Screen 5 - Inventory Planner. Where a shortage becomes an action a planner
// can approve. Each card states what to do, how confident the forecast behind
// it is, and what it assumes - before it asks for approval.

import { useMemo, useState } from "react";

import { Card, ConfidenceBadge, ErrorNote, Pill, Spinner } from "../components/ui";
import type { ActionType, RecommendationCard } from "../lib/api";
import { api } from "../lib/api";
import { ACTION_LABEL, money, num, pct } from "../lib/fmt";
import { useAsync } from "../lib/useAsync";

const ACTIONS: (ActionType | "all")[] = ["all", "transfer", "purchase", "expedite", "defer"];

/** Transfer arrows between hubs - the redeploy-before-you-buy story, visually. */
function TransferArrows({ cards }: { cards: RecommendationCard[] }) {
  const transfers = cards.filter((c) => c.action === "transfer" && c.from_hub_id);
  const pairs = useMemo(() => {
    const m = new Map<string, { from: string; to: string; qty: number; parts: Set<string> }>();
    for (const t of transfers) {
      const k = `${t.from_hub_id}->${t.to_hub_id}`;
      const e = m.get(k) ?? { from: t.from_hub_id!, to: t.to_hub_id, qty: 0, parts: new Set() };
      e.qty += t.qty;
      e.parts.add(t.part_no);
      m.set(k, e);
    }
    return [...m.values()].sort((a, b) => b.qty - a.qty).slice(0, 10);
  }, [transfers]);

  if (!pairs.length)
    return (
      <p className="text-[12px]" style={{ color: "var(--ink-2)" }}>
        No transfers recommended in this view.
      </p>
    );

  const maxQty = Math.max(...pairs.map((p) => p.qty));
  return (
    <ul className="flex flex-col gap-1.5">
      {pairs.map((p) => (
        <li key={`${p.from}-${p.to}`} className="text-[11px]">
          <div className="flex items-center justify-between gap-2">
            <span className="num">
              {p.from} <span style={{ color: "var(--muted)" }}>&rarr;</span>{" "}
              <strong>{p.to}</strong>
            </span>
            <span className="num shrink-0" style={{ color: "var(--ink-2)" }}>
              {num(p.qty, 0)} units &middot; {p.parts.size} part{p.parts.size > 1 ? "s" : ""}
            </span>
          </div>
          <div className="mt-1 h-1.5 overflow-hidden rounded" style={{ background: "var(--grid)" }}>
            <div
              className="h-full rounded"
              style={{ width: `${(p.qty / maxQty) * 100}%`, background: "var(--forecast)" }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function InventoryPlanner({
  hub,
  setHub,
}: {
  hub: string | null;
  setHub: (h: string | null) => void;
}) {
  const [action, setAction] = useState<ActionType | "all">("all");
  const [decided, setDecided] = useState<Record<string, "approved" | "rejected">>({});
  const [openCard, setOpenCard] = useState<string | null>(null);

  const { data, error, loading } = useAsync(
    () => api.recommendations(hub ?? undefined, action === "all" ? undefined : action, 200),
    [hub, action],
  );

  if (error) return <ErrorNote error={error} />;
  if (!data) return <Spinner label="Loading recommendations" />;

  const shortages = data.positions.filter((p) => p.shortfall_vs_p90 > 0).slice(0, 12);
  const excess = data.positions.filter((p) => p.excess_units > 0).slice(0, 10);

  return (
    <div className={loading ? "opacity-60" : ""}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="flex overflow-hidden rounded" style={{ border: "1px solid var(--line)" }}>
          {ACTIONS.map((a, i) => (
            <button
              key={a}
              type="button"
              onClick={() => setAction(a)}
              className="px-2.5 py-1 text-[11px] capitalize"
              style={{
                background: action === a ? "var(--plane)" : "transparent",
                fontWeight: action === a ? 600 : 400,
                borderLeft: i ? "1px solid var(--line)" : undefined,
              }}
            >
              {a === "all" ? "All actions" : ACTION_LABEL[a]}
            </button>
          ))}
        </div>
        {hub ? (
          <button
            type="button"
            onClick={() => setHub(null)}
            className="rounded px-2 py-0.5 text-[11px]"
            style={{ border: "1px solid var(--line)", background: "var(--forecast-wash)" }}
          >
            hub: {hub} &times;
          </button>
        ) : (
          <Pill>all hubs</Pill>
        )}
        <Pill>{data.cards.length} cards</Pill>
        <Pill title="Transfers chosen where expediting was the next-best option">
          {data.expedites_avoided} expedites avoided
        </Pill>
        <Pill>{money(data.total_excess_value)} excess identified</Pill>
      </div>

      <div className="grid gap-3 lg:grid-cols-[1.4fr_1fr]">
        {/* ---- recommendation inbox ---- */}
        <Card
          title="Recommendation inbox"
          subtitle="Each card carries the action, the confidence and the assumptions"
          bodyClass="p-0"
        >
          <ul className="max-h-[620px] overflow-y-auto">
            {data.cards.map((c) => {
              const state = decided[c.rec_id];
              const open = openCard === c.rec_id;
              return (
                <li key={c.rec_id} style={{ borderTop: "1px solid var(--line)" }}>
                  <div className="px-3 py-2.5">
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className="rounded px-1.5 py-0.5 text-[10px] font-semibold tracking-wide uppercase"
                        style={{
                          background:
                            c.action === "transfer"
                              ? "var(--forecast-wash)"
                              : c.action === "expedite"
                                ? "var(--urgent-wash)"
                                : c.action === "defer"
                                  ? "var(--plane)"
                                  : "var(--planned-wash)",
                        }}
                      >
                        {ACTION_LABEL[c.action]}
                      </span>
                      <span className="num text-[12px] font-semibold">
                        {num(c.qty, 0)} &times; {c.part_no}
                      </span>
                      <span className="num text-[11px]" style={{ color: "var(--ink-2)" }}>
                        {c.from_hub_id ? `${c.from_hub_id} → ` : ""}
                        {c.to_hub_id}
                      </span>
                      <span className="ml-auto flex items-center gap-2">
                        <span className="num text-[11px]" style={{ color: "var(--ink-2)" }}>
                          need in {num(c.need_in_days, 1)} d
                        </span>
                        <ConfidenceBadge level={c.confidence} />
                      </span>
                    </div>

                    <p className="mt-1.5 text-[12px]" style={{ color: "var(--ink)" }}>
                      {c.why}
                    </p>

                    <div className="mt-2 flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setOpenCard(open ? null : c.rec_id)}
                        className="rounded px-2 py-1 text-[11px]"
                        style={{ border: "1px solid var(--line)" }}
                      >
                        {open ? "Hide evidence" : `Evidence (${c.evidence.length})`}
                      </button>
                      {state ? (
                        <span
                          className="rounded px-2 py-1 text-[11px] font-medium"
                          style={{ background: "var(--plane)" }}
                        >
                          {state === "approved" ? "Approved - draft created" : "Rejected"}
                        </span>
                      ) : (
                        <>
                          <button
                            type="button"
                            onClick={() =>
                              setDecided((d) => ({ ...d, [c.rec_id]: "approved" }))
                            }
                            className="rounded px-2.5 py-1 text-[11px] font-medium"
                            style={{ background: "var(--ink)", color: "var(--surface)" }}
                          >
                            Approve
                          </button>
                          <button
                            type="button"
                            onClick={() =>
                              setDecided((d) => ({ ...d, [c.rec_id]: "rejected" }))
                            }
                            className="rounded px-2 py-1 text-[11px]"
                            style={{ border: "1px solid var(--line)" }}
                          >
                            Reject
                          </button>
                        </>
                      )}
                      {c.avoided_action && (
                        <span className="ml-auto text-[11px]" style={{ color: "var(--ink-2)" }}>
                          avoids {ACTION_LABEL[c.avoided_action].toLowerCase()} &middot; saves{" "}
                          <span className="num">{money(Math.abs(c.cost_delta))}</span>
                        </span>
                      )}
                    </div>

                    {open && (
                      <div className="mt-2 grid gap-3 md:grid-cols-2">
                        <div>
                          <h5 className="mb-1 text-[10px] font-semibold tracking-wide uppercase" style={{ color: "var(--muted)" }}>
                            Evidence
                          </h5>
                          <ul className="flex flex-col gap-1 text-[11px]">
                            {c.evidence.map((e, i) => (
                              <li key={i} className="flex gap-1.5">
                                <span aria-hidden style={{ color: "var(--muted)" }}>
                                  &bull;
                                </span>
                                <span className="num" style={{ color: "var(--ink-2)" }}>
                                  {e}
                                </span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <h5 className="mb-1 text-[10px] font-semibold tracking-wide uppercase" style={{ color: "var(--muted)" }}>
                            Assumptions
                          </h5>
                          <ul className="flex flex-col gap-1 text-[11px]">
                            {c.assumptions.map((a, i) => (
                              <li key={i}>
                                <span className="font-medium">{a.label}:</span>{" "}
                                <span style={{ color: "var(--ink-2)" }}>{a.text}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        </Card>

        {/* ---- right column: positions, transfers, excess ---- */}
        <div className="flex flex-col gap-3">
          <Card
            title="Transfer map"
            subtitle="Redeployment is checked before procurement, so these come first"
          >
            <TransferArrows cards={data.cards} />
          </Card>

          <Card
            title="On hand vs reorder point"
            subtitle="Largest shortfalls against the 90-day P90 need"
            bodyClass="p-0"
          >
            <table className="w-full text-[11px]">
              <thead className="text-left text-[10px]" style={{ color: "var(--muted)" }}>
                <tr>
                  <th className="px-3 py-1.5 font-normal">Part / hub</th>
                  <th className="px-2 py-1.5 text-right font-normal">Avail.</th>
                  <th className="px-2 py-1.5 text-right font-normal">ROP</th>
                  <th className="px-2 py-1.5 text-right font-normal">P90 need</th>
                  <th className="px-3 py-1.5 text-right font-normal">Lead</th>
                </tr>
              </thead>
              <tbody>
                {shortages.map((p) => (
                  <tr key={`${p.part_no}@${p.hub_id}`} style={{ borderTop: "1px solid var(--line)" }}>
                    <td className="px-3 py-1.5">
                      <div className="num font-medium">{p.part_no}</div>
                      <div style={{ color: "var(--ink-2)" }}>
                        {p.hub_id} &middot; class {p.criticality} &middot;{" "}
                        {pct(p.service_level_target, 0)} target
                      </div>
                    </td>
                    <td className="num px-2 py-1.5 text-right">{num(p.available, 0)}</td>
                    <td className="num px-2 py-1.5 text-right">{num(p.reorder_point, 1)}</td>
                    <td className="num px-2 py-1.5 text-right font-medium">
                      {num(p.projected_need_p90, 1)}
                    </td>
                    <td className="num px-3 py-1.5 text-right">{num(p.lead_time_mean_d, 0)} d</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card
            title="Excess stock"
            subtitle="Flagged before any new buy is suggested anywhere"
            bodyClass="p-0"
          >
            <table className="w-full text-[11px]">
              <thead className="text-left text-[10px]" style={{ color: "var(--muted)" }}>
                <tr>
                  <th className="px-3 py-1.5 font-normal">Part / hub</th>
                  <th className="px-2 py-1.5 text-right font-normal">Excess</th>
                  <th className="px-3 py-1.5 text-right font-normal">Value</th>
                </tr>
              </thead>
              <tbody>
                {excess.map((p) => (
                  <tr key={`x-${p.part_no}@${p.hub_id}`} style={{ borderTop: "1px solid var(--line)" }}>
                    <td className="px-3 py-1.5">
                      <span className="num font-medium">{p.part_no}</span>{" "}
                      <span style={{ color: "var(--ink-2)" }}>{p.hub_id}</span>
                    </td>
                    <td className="num px-2 py-1.5 text-right">{num(p.excess_units, 1)}</td>
                    <td className="num px-3 py-1.5 text-right">{money(p.excess_value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      </div>
    </div>
  );
}
