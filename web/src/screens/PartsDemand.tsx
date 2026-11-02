// Screen 4 - Parts Demand. The translate step: predicted failures become part
// numbers and quantities, split by horizon, and every number drills back to the
// assets and failure modes that produced it.

import { useMemo, useState } from "react";

import { DemandFan } from "../components/charts";
import {
  BUCKET_RANGE,
  BucketChip,
  Card,
  ErrorNote,
  Pill,
  Spinner,
  TableToggle,
} from "../components/ui";
import type { Bucket, PartsDemandRow } from "../lib/api";
import { api } from "../lib/api";
import { money, num, pct, titleCase } from "../lib/fmt";
import { useAsync } from "../lib/useAsync";

const BUCKETS: Bucket[] = ["urgent", "planned", "forecast"];

export default function PartsDemand({
  hub,
  setHub,
  onOpenAsset,
}: {
  hub: string | null;
  setHub: (h: string | null) => void;
  onOpenAsset: (assetId: string) => void;
}) {
  const [bucket, setBucket] = useState<Bucket>("urgent");
  const [sel, setSel] = useState<PartsDemandRow | null>(null);
  const [showTable, setShowTable] = useState(false);

  const { data, error, loading } = useAsync(
    () => api.partsDemand(hub ?? undefined, bucket, 400),
    [hub, bucket],
  );

  const sorted = useMemo(
    () => (data?.rows ?? []).slice().sort((a, b) => b.total_p90 - a.total_p90),
    [data],
  );

  const vanKit = useMemo(() => {
    if (!sel || !data) return [];
    const modes = new Set(sel.contributing_failure_modes);
    return data.van_kits.filter((k) => modes.has(k.failure_mode)).slice(0, 8);
  }, [sel, data]);

  if (error) return <ErrorNote error={error} />;
  if (!data) return <Spinner label="Loading parts demand" />;

  return (
    <div className={loading ? "opacity-60" : ""}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        {/* Horizon is the primary filter, so it reads as a segmented control */}
        <div className="flex overflow-hidden rounded" style={{ border: "1px solid var(--line)" }}>
          {BUCKETS.map((b) => (
            <button
              key={b}
              type="button"
              onClick={() => setBucket(b)}
              className="px-2.5 py-1 text-[11px]"
              style={{
                background: bucket === b ? `var(--${b}-wash)` : "transparent",
                fontWeight: bucket === b ? 600 : 400,
                borderLeft: b !== "urgent" ? "1px solid var(--line)" : undefined,
              }}
            >
              <span
                aria-hidden
                className="mr-1.5 inline-block h-2 w-2 rounded-sm align-middle"
                style={{ background: `var(--${b})` }}
              />
              {b === "urgent" ? "Urgent 0-14 d" : b === "planned" ? "Planned 15-90 d" : "Forecast 90+ d"}
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
        <Pill title="Monte Carlo draws behind each distribution">
          {num(sorted[0]?.draws ?? 1000, 0)} draws
        </Pill>
        <div className="ml-auto">
          <TableToggle open={showTable} onToggle={() => setShowTable((v) => !v)} />
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-[1.25fr_1fr]">
        <Card
          title="Demand by part and hub"
          subtitle={`Bar is P50, whisker is P10-P90, over the ${BUCKET_RANGE[bucket]} horizon. Stocking decisions are taken on P90, not the mean.`}
        >
          <DemandFan rows={sorted} onSelect={setSel} />
        </Card>

        <Card
          title={sel ? `${sel.part_no} at ${sel.hub_id}` : "Pick a part to trace it"}
          subtitle={
            sel
              ? sel.description
              : "Click a bar or a table row to see the assets and failure modes behind the number"
          }
        >
          {!sel ? (
            <p className="text-[12px]" style={{ color: "var(--ink-2)" }}>
              Nothing selected. The traceability panel answers the question a planner always asks
              next: <em>why that many?</em>
            </p>
          ) : (
            <div className="flex flex-col gap-3">
              <div className="grid grid-cols-3 gap-2">
                {[
                  [`P50 demand (${BUCKET_RANGE[sel.bucket]})`, num(sel.demand_p50, 1)],
                  [`P90 demand (${BUCKET_RANGE[sel.bucket]})`, num(sel.demand_p90, 1)],
                  ["Scheduled PM", num(sel.pm_demand, 0)],
                  ["Available now", num(sel.available ?? sel.on_hand, 0)],
                  ["Reorder point (lead time)", num(sel.reorder_point, 1)],
                  ["Safety stock", num(sel.safety_stock, 1)],
                ].map(([k, v]) => (
                  <div key={k} className="rounded px-2 py-1.5" style={{ background: "var(--plane)" }}>
                    <div className="text-[10px]" style={{ color: "var(--ink-2)" }}>
                      {k}
                    </div>
                    <div className="num text-[13px] font-semibold">{v}</div>
                  </div>
                ))}
              </div>

              {(sel.shortfall_vs_p90 ?? 0) > 0 && (
                <div
                  className="rounded px-2.5 py-2 text-[11px]"
                  style={{ background: "var(--urgent-wash)" }}
                >
                  Short <strong className="num">{num(sel.shortfall_vs_p90, 1)}</strong> against the
                  <strong> 90-day</strong> P90 need - a longer horizon than the{" "}
                  {BUCKET_RANGE[sel.bucket]} demand above, so the two figures differ by design. A
                  recommendation card exists for this on the Inventory Planner.
                </div>
              )}

              <div>
                <h4 className="mb-1 text-[11px] font-semibold">
                  Contributing components ({sel.contributing.length})
                </h4>
                <ul className="flex flex-col gap-1">
                  {sel.contributing.map((c) => (
                    <li
                      key={c.component_id}
                      className="flex items-center justify-between gap-2 rounded px-2 py-1 text-[11px]"
                      style={{ border: "1px solid var(--line)" }}
                    >
                      <button
                        type="button"
                        className="text-left hover:underline"
                        onClick={() => onOpenAsset(c.asset_id)}
                      >
                        <span className="num font-medium">{c.asset_id}</span>{" "}
                        <span style={{ color: "var(--ink-2)" }}>
                          {titleCase(c.component_type)} / {titleCase(c.failure_mode)}
                        </span>
                      </button>
                      <span className="num shrink-0" style={{ color: "var(--ink-2)" }}>
                        {pct(c.p_contribution, 0)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>

              {vanKit.length > 0 && (
                <div>
                  <h4 className="mb-1 text-[11px] font-semibold">
                    Van kit for these repairs
                    <span className="ml-1 font-normal" style={{ color: "var(--ink-2)" }}>
                      - the first-time-fix lever
                    </span>
                  </h4>
                  <table className="w-full text-[11px]">
                    <thead className="text-left text-[10px]" style={{ color: "var(--muted)" }}>
                      <tr>
                        <th className="py-1 font-normal">Part</th>
                        <th className="py-1 text-right font-normal">Qty</th>
                        <th className="py-1 text-right font-normal">P(used)</th>
                        <th className="py-1 font-normal">Why carried</th>
                      </tr>
                    </thead>
                    <tbody>
                      {vanKit.map((k, i) => (
                        <tr key={`${k.part_no}-${i}`} style={{ borderTop: "1px solid var(--line)" }}>
                          <td className="num py-1">{k.part_no}</td>
                          <td className="num py-1 text-right">{num(k.qty, 1)}</td>
                          <td className="num py-1 text-right">{pct(k.p_used, 0)}</td>
                          <td className="py-1" style={{ color: "var(--ink-2)" }}>
                            {k.reason === "primary" ? "likely the fix" : "cheap, worth carrying"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </Card>
      </div>

      {showTable && (
        <Card
          className="mt-3"
          title="Demand table"
          subtitle="The relief view - every plotted value in text form"
          bodyClass="p-0"
        >
          <div className="max-h-[360px] overflow-auto">
            <table className="w-full text-[11px]">
              <thead
                className="sticky top-0 text-left text-[10px]"
                style={{ background: "var(--surface)", color: "var(--muted)" }}
              >
                <tr>
                  {[
                  "Part", "Description", "Hub", "Horizon",
                  "P10", "P50", "P90", "PM",
                  "Avail.", "Short vs 90d P90", "Unit cost",
                ].map(
                    (h) => (
                      <th key={h} className="px-2 py-1.5 font-normal whitespace-nowrap">
                        {h}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {sorted.map((r) => (
                  <tr
                    key={`${r.part_no}@${r.hub_id}`}
                    className="cursor-pointer hover:brightness-[0.98]"
                    style={{ borderTop: "1px solid var(--line)" }}
                    onClick={() => setSel(r)}
                  >
                    <td className="num px-2 py-1.5 font-medium">{r.part_no}</td>
                    <td className="px-2 py-1.5" style={{ color: "var(--ink-2)" }}>
                      {r.description}
                    </td>
                    <td className="num px-2 py-1.5">{r.hub_id}</td>
                    <td className="px-2 py-1.5">
                      <BucketChip bucket={r.bucket} />
                    </td>
                    <td className="num px-2 py-1.5">{num(r.demand_p10, 1)}</td>
                    <td className="num px-2 py-1.5 font-medium">{num(r.demand_p50, 1)}</td>
                    <td className="num px-2 py-1.5 font-medium">{num(r.demand_p90, 1)}</td>
                    <td className="num px-2 py-1.5">{num(r.pm_demand, 0)}</td>
                    <td className="num px-2 py-1.5">{num(r.available ?? r.on_hand, 0)}</td>
                    <td className="num px-2 py-1.5">
                      {(r.shortfall_vs_p90 ?? 0) > 0 ? num(r.shortfall_vs_p90, 1) : "-"}
                    </td>
                    <td className="num px-2 py-1.5">{money(r.unit_cost)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
