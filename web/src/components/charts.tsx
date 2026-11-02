// Charts. Every one ships a hover layer, a legend whenever two or more series
// are present, and recessive grid/axes. Band fills use the single shared
// --opacity-band so uncertainty looks the same everywhere.

import { useId, useMemo, useState } from "react";
import {
  Area,
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import type { Bucket, HubRisk, PartsDemandRow, SensorSeries } from "../lib/api";
import { days, num, pct, titleCase } from "../lib/fmt";
import { bucketVar, BUCKET_LABEL } from "./ui";

const GRID = "var(--grid)";
const INK = "var(--ink)";
const INK2 = "var(--ink-2)";
const MUTED = "var(--muted)";
const SURFACE = "var(--surface)";

/** Shared tooltip shell, so every chart's hover looks identical. */
function TipBox({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="rounded px-2.5 py-2 text-[11px] shadow-sm"
      style={{ background: SURFACE, border: "1px solid var(--line)", color: INK }}
    >
      {children}
    </div>
  );
}

// --------------------------------------------------------------------------
// BandBar - the P10/P50/P90 mark used in every table row
// --------------------------------------------------------------------------

/**
 * Inline uncertainty band on a shared log scale.
 *
 * A log scale because TTM spans 1 day to 10 years; on a linear axis every
 * urgent item collapses into the first pixel. The P50 is a 2px tick rather
 * than a dot so it reads against the band at small sizes.
 */
export function BandBar({
  p10,
  p50,
  p90,
  bucket,
  beyond = false,
  width = 150,
  max = 3650,
}: {
  p10: number;
  p50: number;
  p90: number;
  bucket: Bucket;
  beyond?: boolean;
  width?: number;
  max?: number;
}) {
  const h = 16;
  const lo = Math.log10(0.7);
  const hi = Math.log10(max);
  const x = (v: number) => {
    const c = Math.min(Math.max(v, 0.7), max);
    return ((Math.log10(c) - lo) / (hi - lo)) * width;
  };
  const x10 = x(p10);
  const x90 = x(p90);
  const x50 = x(p50);
  const color = bucketVar(bucket);
  const title = `P10 ${days(p10)} - P50 ${days(p50, beyond)} - P90 ${days(p90, beyond)}`;

  return (
    <svg width={width} height={h} role="img" aria-label={title}>
      <title>{title}</title>
      {/* 14 / 90-day horizon guides, so a band reads against the buckets */}
      {[14, 90].map((g) => (
        <line key={g} x1={x(g)} x2={x(g)} y1={2} y2={h - 2} stroke={GRID} strokeWidth={1} />
      ))}
      {/* the band: 4px rounded ends */}
      <rect
        x={x10}
        y={h / 2 - 3}
        width={Math.max(x90 - x10, 3)}
        height={6}
        rx={3}
        fill={color}
        fillOpacity={0.28}
      />
      {/* P50 tick, with a 2px surface ring so it survives on top of the band */}
      <line x1={x50} x2={x50} y1={h / 2 - 6} y2={h / 2 + 6} stroke={SURFACE} strokeWidth={4} />
      <line x1={x50} x2={x50} y1={h / 2 - 5} y2={h / 2 + 5} stroke={color} strokeWidth={2} />
    </svg>
  );
}

/** Legend for the BandBar, shown once per table rather than per row. */
export function BandLegend() {
  return (
    <div className="flex flex-wrap items-center gap-3 text-[11px]" style={{ color: INK2 }}>
      <span className="inline-flex items-center gap-1.5">
        <svg width={34} height={10} aria-hidden>
          <rect x={0} y={2} width={34} height={6} rx={3} fill={INK} fillOpacity={0.28} />
          <line x1={17} x2={17} y1={0} y2={10} stroke={INK} strokeWidth={2} />
        </svg>
        P10-P90 band, P50 marked
      </span>
      <span>log scale; ticks at 14 and 90 days</span>
    </div>
  );
}

// --------------------------------------------------------------------------
// Survival curve
// --------------------------------------------------------------------------

export function SurvivalChart({
  survival,
  p10,
  p50,
  p90,
  bucket,
  height = 170,
}: {
  survival: { days: number[]; survival: number[] };
  p10: number;
  p50: number;
  p90: number;
  bucket: Bucket;
  height?: number;
}) {
  const gid = useId().replace(/:/g, "");
  const data = useMemo(
    () => survival.days.map((d, i) => ({ d, s: survival.survival[i] })),
    [survival],
  );
  const color = bucketVar(bucket);
  if (!data.length) return null;

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
          <defs>
            <linearGradient id={`sv-${gid}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.22} />
              <stop offset="100%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={GRID} strokeDasharray="2 3" vertical={false} />
          <XAxis
            dataKey="d"
            type="number"
            tickFormatter={(v: number) => (v >= 365 ? `${(v / 365).toFixed(0)}y` : `${Math.round(v)}d`)}
            stroke={GRID}
            tick={{ fontSize: 11, fill: MUTED }}
            label={{ value: "days from today", position: "insideBottom", offset: -8, fontSize: 11, fill: MUTED }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            stroke={GRID}
            tick={{ fontSize: 11, fill: MUTED }}
            width={38}
          />
          {/* Band edges as guides, so the chart and the table agree visually */}
          <ReferenceLine x={p10} stroke={color} strokeOpacity={0.45} strokeDasharray="3 3" />
          <ReferenceLine x={p90} stroke={color} strokeOpacity={0.45} strokeDasharray="3 3" />
          <ReferenceLine
            x={p50}
            stroke={color}
            strokeWidth={2}
            label={{ value: `P50 ${days(p50)}`, fontSize: 10, fill: INK2, position: "top" }}
          />
          <Area type="monotone" dataKey="s" stroke={color} strokeWidth={2} fill={`url(#sv-${gid})`} />
          <Tooltip
            content={({ active, payload }) =>
              active && payload?.length ? (
                <TipBox>
                  <div className="num font-medium">{days(payload[0].payload.d)}</div>
                  <div style={{ color: INK2 }}>
                    survives: <span className="num">{pct(payload[0].payload.s, 1)}</span>
                  </div>
                  <div style={{ color: INK2 }}>
                    fails by then: <span className="num">{pct(1 - payload[0].payload.s, 1)}</span>
                  </div>
                </TipBox>
              ) : null
            }
          />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="mt-1 text-[11px]" style={{ color: MUTED }}>
        Probability the component is still running after N days. Dashed lines are the calibrated
        P10 and P90 bounds.
      </p>
    </div>
  );
}

// --------------------------------------------------------------------------
// Sensor trend vs twin baseline
// --------------------------------------------------------------------------

export function TrendChart({ series, height = 150 }: { series: SensorSeries; height?: number }) {
  const data = useMemo(
    () =>
      series.timestamps.map((t, i) => ({
        t: new Date(t).getTime(),
        observed: series.observed[i],
        baseline: series.baseline[i] ?? null,
      })),
    [series],
  );
  const hasBaseline = series.baseline?.some((v) => v !== null && v !== undefined);

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <h4 className="text-[12px] font-medium">
          {series.display_name}{" "}
          <span style={{ color: MUTED }} className="font-normal">
            ({series.unit})
          </span>
        </h4>
        {/* Two series -> direct labels rather than a legend box */}
        {hasBaseline && (
          <div className="flex items-center gap-3 text-[10px]" style={{ color: INK2 }}>
            <span className="inline-flex items-center gap-1">
              <svg width={14} height={8} aria-hidden>
                <line x1={0} y1={4} x2={14} y2={4} stroke="var(--forecast)" strokeWidth={2} />
              </svg>
              observed
            </span>
            <span className="inline-flex items-center gap-1">
              <svg width={14} height={8} aria-hidden>
                <line
                  x1={0}
                  y1={4}
                  x2={14}
                  y2={4}
                  stroke={MUTED}
                  strokeWidth={2}
                  strokeDasharray="3 2"
                />
              </svg>
              healthy baseline
            </span>
          </div>
        )}
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="2 3" vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            scale="time"
            tickFormatter={(v: number) =>
              new Date(v).toLocaleDateString(undefined, { day: "numeric", month: "short" })
            }
            stroke={GRID}
            tick={{ fontSize: 10, fill: MUTED }}
            minTickGap={40}
          />
          <YAxis
            stroke={GRID}
            tick={{ fontSize: 10, fill: MUTED }}
            width={44}
            domain={["auto", "auto"]}
          />
          {hasBaseline && (
            <Line
              type="monotone"
              dataKey="baseline"
              stroke={MUTED}
              strokeWidth={2}
              strokeDasharray="3 2"
              dot={false}
              isAnimationActive={false}
            />
          )}
          <Line
            type="monotone"
            dataKey="observed"
            stroke="var(--forecast)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload as { t: number; observed: number | null; baseline: number | null };
              const resid =
                p.observed !== null && p.baseline !== null ? p.observed - p.baseline : null;
              return (
                <TipBox>
                  <div className="num font-medium">
                    {new Date(p.t).toLocaleDateString(undefined, {
                      day: "numeric",
                      month: "short",
                      year: "numeric",
                    })}
                  </div>
                  <div style={{ color: INK2 }}>
                    observed <span className="num">{num(p.observed, 1)}</span> {series.unit}
                  </div>
                  {p.baseline !== null && (
                    <div style={{ color: INK2 }}>
                      baseline <span className="num">{num(p.baseline, 1)}</span>
                    </div>
                  )}
                  {resid !== null && (
                    <div style={{ color: INK }}>
                      residual <span className="num">{resid > 0 ? "+" : ""}{num(resid, 1)}</span>
                    </div>
                  )}
                </TipBox>
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// --------------------------------------------------------------------------
// Demand fan
// --------------------------------------------------------------------------

/**
 * Per-part demand with its P10-P90 range.
 *
 * The bar is P50 and the whisker is the band, because stocking decisions are
 * taken on P90 - showing the mean alone would hide exactly the spread that the
 * safety-stock calculation consumes.
 */
export function DemandFan({
  rows,
  height = 260,
  onSelect,
}: {
  rows: PartsDemandRow[];
  height?: number;
  onSelect?: (r: PartsDemandRow) => void;
}) {
  const data = useMemo(
    () =>
      rows.slice(0, 12).map((r) => ({
        key: `${r.part_no}@${r.hub_id}`,
        label: r.part_no,
        hub: r.hub_id,
        p10: r.demand_p10,
        p50: r.demand_p50,
        p90: r.demand_p90,
        lowSpan: Math.max(r.demand_p50 - r.demand_p10, 0),
        highSpan: Math.max(r.demand_p90 - r.demand_p50, 0),
        pm: r.pm_demand,
        onHand: r.available ?? r.on_hand ?? 0,
        bucket: r.bucket,
        row: r,
      })),
    [rows],
  );
  if (!data.length) return <p className="p-4 text-[12px]" style={{ color: INK2 }}>No demand lines.</p>;

  const barClick = ((d: unknown) => {
    const row = (d as { row?: PartsDemandRow } | null)?.row;
    if (row) onSelect?.(row);
  }) as React.ComponentProps<typeof Bar>["onClick"];

  const barShape = ((props: unknown) => {
    const { x, y, width, height, payload } = props as {
      x: number; y: number; width: number; height: number; payload: { bucket: Bucket };
    };
    return (
      <rect
        x={x}
        y={y}
        width={width}
        height={Math.max(height, 0)}
        rx={4}
        fill={bucketVar(payload.bucket)}
        stroke={SURFACE}
        strokeWidth={2}
      />
    );
  }) as React.ComponentProps<typeof Bar>["shape"];


  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-3 text-[11px]" style={{ color: INK2 }}>
        <span className="inline-flex items-center gap-1.5">
          <svg width={16} height={10} aria-hidden>
            <rect x={6} y={1} width={4} height={8} rx={1} fill={INK} fillOpacity={0.9} />
          </svg>
          P50 demand
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width={16} height={10} aria-hidden>
            <line x1={8} y1={0} x2={8} y2={10} stroke={INK} strokeWidth={2} strokeOpacity={0.35} />
          </svg>
          P10-P90 range
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width={16} height={10} aria-hidden>
            <line x1={0} y1={5} x2={16} y2={5} stroke={MUTED} strokeWidth={2} strokeDasharray="3 2" />
          </svg>
          available stock
        </span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 6, right: 12, bottom: 44, left: 4 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="2 3" vertical={false} />
          <XAxis
            dataKey="label"
            stroke={GRID}
            tick={{ fontSize: 10, fill: MUTED }}
            angle={-35}
            textAnchor="end"
            interval={0}
            height={50}
          />
          <YAxis stroke={GRID} tick={{ fontSize: 10, fill: MUTED }} width={34} allowDecimals={false} />
          {/* Stacked invisible base + spans renders the range as a whisker */}
          <Bar dataKey="p10" stackId="band" fill="transparent" isAnimationActive={false} />
          <Bar
            dataKey="lowSpan"
            stackId="band"
            isAnimationActive={false}
            barSize={3}
            fill={INK}
            fillOpacity={0.3}
          />
          <Bar
            dataKey="highSpan"
            stackId="band"
            isAnimationActive={false}
            barSize={3}
            fill={INK}
            fillOpacity={0.3}
          />
          <Bar
            dataKey="p50"
            isAnimationActive={false}
            barSize={14}
            radius={[4, 4, 0, 0]}
            onClick={barClick}
            // Colour by horizon, which is what the fill means on every screen
            shape={barShape}
          />
          <Line
            type="stepAfter"
            dataKey="onHand"
            stroke={MUTED}
            strokeWidth={2}
            strokeDasharray="3 2"
            dot={false}
            isAnimationActive={false}
          />
          <Tooltip
            cursor={{ fill: "var(--grid)", fillOpacity: 0.35 }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload as (typeof data)[number];
              const short = d.p90 - d.onHand;
              return (
                <TipBox>
                  <div className="font-medium">{d.label}</div>
                  <div style={{ color: INK2 }}>
                    {d.hub} - {BUCKET_LABEL[d.bucket]}
                  </div>
                  <div className="mt-1 num">
                    P50 {num(d.p50, 1)} - P90 {num(d.p90, 1)}
                  </div>
                  {d.pm > 0 && (
                    <div className="num" style={{ color: INK2 }}>
                      + {num(d.pm, 0)} scheduled PM
                    </div>
                  )}
                  <div className="num" style={{ color: INK2 }}>
                    available {num(d.onHand, 0)}
                  </div>
                  {short > 0 && (
                    <div className="num mt-1" style={{ color: "var(--urgent)" }}>
                      short {num(short, 1)} vs P90
                    </div>
                  )}
                </TipBox>
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// --------------------------------------------------------------------------
// Hub risk map
// --------------------------------------------------------------------------

const SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"];

/**
 * Hub scatter on raw lat/lon.
 *
 * Deliberately not a tiled basemap: tiles are an external network dependency,
 * and a demo must not go blank because a tile server is unreachable. Relative
 * hub geography is all this view needs to convey.
 *
 * Risk is sequential - one hue, light to dark - because it is a magnitude.
 */
export function HubMap({
  hubs,
  selected,
  onSelect,
  height = 280,
}: {
  hubs: HubRisk[];
  selected?: string | null;
  onSelect?: (hubId: string | null) => void;
  height?: number;
}) {
  const data = hubs.map((h) => ({ ...h, z: Math.max(h.urgent_jobs, 1) }));
  const color = (r: number) => SEQ[Math.min(Math.floor(r * SEQ.length), SEQ.length - 1)];

  const hubClick = ((d: unknown) => {
    const id = (d as { hub_id?: string } | null)?.hub_id ?? null;
    onSelect?.(id === selected ? null : id);
  }) as React.ComponentProps<typeof Scatter>["onClick"];

  const hubShape = ((props: unknown) => {
    const { cx, cy, payload, size } = props as {
      cx: number; cy: number; payload: HubRisk; size: number;
    };
    const r = Math.max(Math.sqrt((size ?? 120) / Math.PI), 7);
    const isSel = payload.hub_id === selected;
    return (
      <g style={{ cursor: "pointer" }}>
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill={color(payload.risk_score)}
          stroke={isSel ? INK : SURFACE}
          strokeWidth={2}
        />
        <text
          x={cx}
          y={cy - r - 4}
          textAnchor="middle"
          fontSize={10}
          fill={isSel ? INK : INK2}
          fontWeight={isSel ? 600 : 400}
        >
          {payload.name}
        </text>
      </g>
    );
  }) as React.ComponentProps<typeof Scatter>["shape"];


  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 10, right: 16, bottom: 24, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="2 3" />
          <XAxis
            dataKey="lon"
            type="number"
            domain={["dataMin - 2", "dataMax + 2"]}
            tickFormatter={(v: number) => `${v.toFixed(0)}E`}
            stroke={GRID}
            tick={{ fontSize: 10, fill: MUTED }}
          />
          <YAxis
            dataKey="lat"
            type="number"
            domain={["dataMin - 2", "dataMax + 2"]}
            tickFormatter={(v: number) => `${v.toFixed(0)}N`}
            stroke={GRID}
            tick={{ fontSize: 10, fill: MUTED }}
            width={38}
          />
          <ZAxis dataKey="z" range={[90, 420]} />
          <Scatter data={data} isAnimationActive={false} onClick={hubClick} shape={hubShape} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const h = payload[0].payload as HubRisk;
              return (
                <TipBox>
                  <div className="font-medium">
                    {h.name} <span style={{ color: INK2 }}>({h.region})</span>
                  </div>
                  <div className="num mt-1" style={{ color: INK2 }}>
                    {h.urgent_jobs} urgent jobs - {h.assets_at_risk} assets at risk
                  </div>
                  <div className="num" style={{ color: INK2 }}>
                    {h.projected_stockouts} projected stockouts
                  </div>
                  <div className="mt-1 text-[10px]" style={{ color: MUTED }}>
                    click to filter the whole screen
                  </div>
                </TipBox>
              );
            }}
          />
        </ScatterChart>
      </ResponsiveContainer>
      <div className="mt-1 flex items-center justify-between text-[11px]" style={{ color: INK2 }}>
        <span className="inline-flex items-center gap-1.5">
          urgent jobs
          <span aria-hidden className="inline-flex items-center gap-[3px]">
            {SEQ.map((c) => (
              <span key={c} className="inline-block h-2.5 w-4 rounded-sm" style={{ background: c }} />
            ))}
          </span>
          more
        </span>
        <span style={{ color: MUTED }}>circle area = urgent job count</span>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------
// Risk heatmap (region x component)
// --------------------------------------------------------------------------

export function RiskHeatmap({ heatmap }: { heatmap: Record<string, Record<string, number>> }) {
  const [showTable, setShowTable] = useState(false);
  const regions = Object.keys(heatmap).sort();
  const comps = useMemo(() => {
    const s = new Set<string>();
    for (const r of regions) for (const c of Object.keys(heatmap[r])) s.add(c);
    return [...s].sort();
  }, [heatmap, regions]);
  if (!regions.length) return null;

  const cell = (v: number) => SEQ[Math.min(Math.floor(v * SEQ.length), SEQ.length - 1)];

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full border-separate" style={{ borderSpacing: "2px" }}>
          <thead>
            <tr>
              <th />
              {comps.map((c) => (
                <th
                  key={c}
                  className="pb-1 text-left text-[10px] font-normal"
                  style={{ color: MUTED, minWidth: 54 }}
                >
                  {titleCase(c).slice(0, 12)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {regions.map((r) => (
              <tr key={r}>
                <th className="pr-2 text-right text-[11px] font-medium whitespace-nowrap">{r}</th>
                {comps.map((c) => {
                  const v = heatmap[r][c];
                  return (
                    <td
                      key={c}
                      title={`${r} / ${titleCase(c)}: P(fail within 90 d) ${pct(v, 1)}`}
                      className="num rounded text-center text-[10px]"
                      style={{
                        background: v === undefined ? "transparent" : cell(v),
                        color: v > 0.55 ? "#fff" : INK,
                        height: 26,
                      }}
                    >
                      {v === undefined ? "" : pct(v, 0)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex items-center justify-between text-[11px]" style={{ color: INK2 }}>
        <span>Mean P(fail within 90 days). Darker is higher risk.</span>
        <button
          type="button"
          onClick={() => setShowTable((v) => !v)}
          className="rounded px-2 py-1"
          style={{ border: "1px solid var(--line)" }}
        >
          {showTable ? "Hide values" : "View as table"}
        </button>
      </div>
      {showTable && (
        <pre
          className="num mt-2 overflow-x-auto rounded p-2 text-[10px]"
          style={{ background: "var(--plane)", color: INK2 }}
        >
          {regions
            .map((r) => `${r}: ` + comps.map((c) => `${c}=${pct(heatmap[r][c], 1)}`).join("  "))
            .join("\n")}
        </pre>
      )}
    </div>
  );
}
