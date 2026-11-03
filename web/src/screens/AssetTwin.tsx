// Screen 3 - Asset Twin. Everything the brief asks about a single prediction on
// one page: the health state, the TTM band, the drivers behind it, and the
// assumptions it rests on.

import { lazy, Suspense, useState } from "react";

import { BandBar, SurvivalChart, TrendChart } from "../components/charts";
import FaultStatePanel from "../components/faultstate";
const Twin3D = lazy(() => import("../components/twin3d"));
import WhatIfPanel from "../components/whatif";
import { BucketChip, Card, ConfidenceBadge, ErrorNote, Pill, Spinner } from "../components/ui";
import { api } from "../lib/api";
import { days, num, pct, titleCase } from "../lib/fmt";
import { useAsync } from "../lib/useAsync";

/** Vapour-compression schematic, coloured by each component's health index. */
function CycleSchematic({ health }: { health: Record<string, number> }) {
  // Health is in [0,1] where 1 is healthy; map onto the sequential ramp so a
  // degrading part darkens rather than changing hue.
  const SEQ = ["#104281", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"];
  const col = (k: string) => {
    const v = health[k];
    if (v === undefined) return "var(--grid)";
    return SEQ[Math.min(Math.floor(v * SEQ.length), SEQ.length - 1)];
  };
  const boxes: { k: string; label: string; x: number; y: number }[] = [
    { k: "compressor", label: "Compressor", x: 12, y: 74 },
    { k: "condenser_coil", label: "Condenser", x: 112, y: 12 },
    { k: "eev", label: "EEV / TXV", x: 212, y: 74 },
    { k: "evaporator", label: "Evaporator", x: 112, y: 128 },
  ];
  return (
    <div>
      <svg viewBox="0 0 300 176" className="w-full" role="img" aria-label="Refrigerant cycle health">
        <title>Refrigerant cycle, shaded by component health</title>
        {/* circuit */}
        <path
          d="M52 86 L52 32 L152 32 M212 32 L252 32 L252 86 M252 118 L252 148 L152 148 M112 148 L52 148 L52 110"
          fill="none"
          stroke="var(--axis)"
          strokeWidth={2}
        />
        {boxes.map((b) => (
          <g key={b.k}>
            <rect
              x={b.x}
              y={b.y}
              width={76}
              height={36}
              rx={5}
              fill={col(b.k)}
              stroke="var(--surface)"
              strokeWidth={2}
            />
            <text
              x={b.x + 38}
              y={b.y + 22}
              textAnchor="middle"
              fontSize={9}
              fill={health[b.k] !== undefined && health[b.k] < 0.45 ? "#fff" : "var(--ink)"}
            >
              {b.label}
            </text>
          </g>
        ))}
      </svg>
      <div className="mt-1 flex items-center justify-between text-[11px]" style={{ color: "var(--ink-2)" }}>
        <span className="inline-flex items-center gap-1.5">
          degraded
          <span aria-hidden className="inline-flex gap-[3px]">
            {SEQ.map((c) => (
              <span key={c} className="inline-block h-2.5 w-4 rounded-sm" style={{ background: c }} />
            ))}
          </span>
          healthy
        </span>
      </div>
    </div>
  );
}

export default function AssetTwin({
  assetId,
  onBack,
}: {
  assetId: string;
  onBack: () => void;
}) {
  const { data, error, loading } = useAsync(() => api.asset(assetId), [assetId]);
  const [sel, setSel] = useState(0);

  if (error) return <ErrorNote error={error} />;
  if (!data) return <Spinner label={`Loading ${assetId}`} />;

  const comp = data.components[Math.min(sel, data.components.length - 1)];

  return (
    <div className={loading ? "opacity-60" : ""}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={onBack}
          className="rounded px-2 py-1 text-[11px]"
          style={{ border: "1px solid var(--line)" }}
        >
          &larr; Back
        </button>
        <h1 className="text-[15px] font-semibold">{data.asset_label}</h1>
        <Pill>{data.model}</Pill>
        <Pill>{num(data.capacity_kw, 0)} kW</Pill>
        <Pill>class {data.criticality}</Pill>
        <Pill title="Climate zone drives thermal stress in the degradation model">
          zone {data.climate_zone}
        </Pill>
        <span className="text-[11px]" style={{ color: "var(--ink-2)" }}>
          {data.site_name} &middot; {data.hub_id}
        </span>
      </div>

      <div className="grid gap-3 lg:grid-cols-[320px_1fr]">
        {/* ---- left: twin + component list ---- */}
        <div className="flex flex-col gap-3">
          <Card
            title="Digital twin"
            subtitle="The refrigerant circuit of this machine, shaded by predicted health. Drag to rotate; click a part to select it."
          >
            <Suspense
              fallback={
                <div
                  className="flex items-center justify-center rounded text-[11px]"
                  style={{ height: 300, background: "var(--plane)", color: "var(--ink-2)" }}
                >
                  Loading the 3D view...
                </div>
              }
            >
            <Twin3D
              health={data.twin.component_health}
              selectedType={comp?.component_type ?? null}
              onSelectType={(t) => {
                const i = data.components.findIndex((c) => c.component_type === t);
                if (i >= 0) setSel(i);
              }}
              fallback={<CycleSchematic health={data.twin.component_health} />}
            />
            </Suspense>
            <dl className="num mt-3 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
              {[
                ["Suction", data.twin.suction_p_kpa, "kPa"],
                ["Discharge", data.twin.discharge_p_kpa, "kPa"],
                ["Superheat", data.twin.superheat_k, "K"],
                ["COP", data.twin.cop, ""],
                ["Peak ambient", data.twin.ambient_t_max, "C"],
              ].map(([k, v, u]) => (
                <div key={String(k)} className="flex justify-between gap-2">
                  <dt style={{ color: "var(--ink-2)" }}>{k}</dt>
                  <dd className="font-medium">
                    {num(v as number, 1)} {u as string}
                  </dd>
                </div>
              ))}
            </dl>
          </Card>

          <Card title="Components by predicted TTM" bodyClass="p-0">
            <ul className="max-h-[320px] overflow-y-auto">
              {data.components.map((c, i) => (
                <li key={c.component_id} style={{ borderTop: i ? "1px solid var(--line)" : undefined }}>
                  <button
                    type="button"
                    onClick={() => setSel(i)}
                    className="w-full px-3 py-2 text-left hover:brightness-[0.98]"
                    style={{
                      background: i === sel ? "var(--plane)" : undefined,
                      borderLeft: i === sel ? "3px solid var(--ink)" : "3px solid transparent",
                    }}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[12px] font-medium">{titleCase(c.component_type)}</span>
                      <BucketChip bucket={c.bucket} />
                    </div>
                    <div className="mt-1 flex items-center gap-2">
                      <BandBar
                        p10={c.p10_days}
                        p50={c.p50_days}
                        p90={c.p90_days}
                        bucket={c.bucket}
                        beyond={c.beyond_horizon}
                        width={130}
                      />
                      <span className="num text-[11px] font-medium">
                        {days(c.p50_days, c.beyond_horizon)}
                      </span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </Card>
        </div>

        {/* ---- right: the selected prediction, in full ---- */}
        <div className="flex flex-col gap-3">
          <Card
            title={`${titleCase(comp.component_type)} - time to maintenance`}
            subtitle="Survival curve with the calibrated P10-P90 band"
            right={
              <div className="flex items-center gap-2">
                <BucketChip bucket={comp.bucket} range />
                <ConfidenceBadge level={comp.confidence} reasons={comp.confidence_reasons} />
              </div>
            }
          >
            <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                ["P10 (early)", days(comp.p10_days)],
                ["P50 (median)", days(comp.p50_days, comp.beyond_horizon)],
                ["P90 (late)", days(comp.p90_days, comp.beyond_horizon)],
                ["P(fail <= 14 d)", pct(comp.p_fail_14d, 1)],
              ].map(([k, v]) => (
                <div key={k} className="rounded px-2.5 py-2" style={{ background: "var(--plane)" }}>
                  <div className="text-[10px]" style={{ color: "var(--ink-2)" }}>
                    {k}
                  </div>
                  <div className="num mt-0.5 text-[15px] font-semibold">{v}</div>
                </div>
              ))}
            </div>
            <SurvivalChart
              survival={comp.survival}
              p10={comp.p10_days}
              p50={comp.p50_days}
              p90={comp.p90_days}
              bucket={comp.bucket}
            />
          </Card>

          <div className="grid gap-3 md:grid-cols-2">
            <Card title="Why this prediction" subtitle="SHAP contribution, in log-days">
              <ul className="flex flex-col gap-2">
                {comp.drivers.map((d) => {
                  const mag = Math.min(Math.abs(d.shap_value) / 1.5, 1);
                  const shortens = d.direction === "shortens_life";
                  return (
                    <li key={d.feature}>
                      <div className="flex items-baseline justify-between gap-2 text-[11px]">
                        <span>{d.display_name}</span>
                        <span className="num shrink-0" style={{ color: "var(--ink-2)" }}>
                          {d.shap_value > 0 ? "+" : ""}
                          {num(d.shap_value, 2)}
                        </span>
                      </div>
                      {/* Direction is carried by the arrow and the word, not by hue */}
                      <div className="mt-1 flex items-center gap-1.5">
                        <span aria-hidden style={{ color: "var(--ink-2)" }} className="text-[10px]">
                          {shortens ? "▼" : "▲"}
                        </span>
                        <div
                          className="h-1.5 flex-1 overflow-hidden rounded"
                          style={{ background: "var(--grid)" }}
                        >
                          <div
                            className="h-full rounded"
                            style={{
                              width: `${Math.max(mag * 100, 4)}%`,
                              background: shortens ? "var(--urgent)" : "var(--forecast)",
                            }}
                          />
                        </div>
                        <span className="text-[10px]" style={{ color: "var(--ink-2)" }}>
                          {shortens ? "shortens life" : "extends life"}
                        </span>
                      </div>
                    </li>
                  );
                })}
              </ul>
            </Card>

            <Card
              title="What this assumes"
              subtitle="Required by the brief - stated in plain words, from the actual feature values"
            >
              <ul className="flex flex-col gap-2 text-[11px]">
                {comp.assumptions.map((a, i) => (
                  <li key={i} className="rounded px-2.5 py-2" style={{ background: "var(--plane)" }}>
                    <div className="font-medium">{a.label}</div>
                    <div style={{ color: "var(--ink-2)" }}>{a.text}</div>
                    {a.basis && (
                      <div className="mt-0.5 text-[10px]" style={{ color: "var(--muted)" }}>
                        basis: {a.basis}
                      </div>
                    )}
                  </li>
                ))}
              </ul>
              <div className="mt-2 text-[10px]" style={{ color: "var(--muted)" }}>
                Data quality {pct(comp.data_quality, 0)} &middot; band width{" "}
                {days(comp.interval_width_days)}
              </div>
            </Card>
          </div>

          <FaultStatePanel state={data.fault_state} />

          <WhatIfPanel assetId={data.asset_id} />

          <Card
            title="Sensor trends against the healthy baseline"
            subtitle={data.sensors[0]?.baseline_source}
          >
            <div className="grid gap-4 md:grid-cols-2">
              {data.sensors.slice(0, 4).map((s) => (
                <TrendChart key={s.signal} series={s} />
              ))}
            </div>
          </Card>

          <Card title="Service history" bodyClass="p-0">
            <div className="max-h-[220px] overflow-y-auto">
              <table className="w-full text-[11px]">
                <thead
                  className="sticky top-0 text-left text-[10px]"
                  style={{ background: "var(--surface)", color: "var(--muted)" }}
                >
                  <tr>
                    <th className="px-3 py-1.5 font-normal">Date</th>
                    <th className="px-2 py-1.5 font-normal">Type</th>
                    <th className="px-2 py-1.5 font-normal">Failure mode</th>
                    <th className="px-2 py-1.5 font-normal">Parts</th>
                    <th className="px-3 py-1.5 font-normal">Note</th>
                  </tr>
                </thead>
                <tbody>
                  {data.history.map((h) => (
                    <tr key={h.wo_id} style={{ borderTop: "1px solid var(--line)" }}>
                      <td className="num px-3 py-1.5 whitespace-nowrap">
                        {new Date(h.ts).toLocaleDateString(undefined, {
                          day: "numeric",
                          month: "short",
                          year: "2-digit",
                        })}
                      </td>
                      <td className="px-2 py-1.5">{h.type}</td>
                      <td className="px-2 py-1.5">{h.failure_mode ? titleCase(h.failure_mode) : "-"}</td>
                      <td className="num px-2 py-1.5">{h.parts_used?.join(", ") || "-"}</td>
                      <td className="px-3 py-1.5" style={{ color: "var(--ink-2)" }}>
                        {h.notes}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
