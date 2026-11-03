// What-if panel. The one place in the app where the model runs live, so it is
// the one place the audience sees the chain move: conditions -> twin -> TTM
// band -> horizon bucket -> parts.
//
// Two design rules carry most of the weight here:
//
//   1. The sliders move *conditions*, never predictions. There is no control
//      on this panel that can push a TTM directly, which is what makes the
//      response evidence rather than decoration.
//   2. Before and after are always shown together. A single set of numbers
//      that silently changed under the slider would be unreadable; the point
//      is the delta, so the delta is what gets the emphasis.

import { useCallback, useEffect, useRef, useState } from "react";

import type { Bucket, WhatIfResponse } from "../lib/api";
import { api } from "../lib/api";
import { days, num, pct, titleCase } from "../lib/fmt";
import { BUCKET_LABEL, BucketChip, Card, Pill, bucketVar, bucketWash } from "./ui";

const BUCKETS: Bucket[] = ["urgent", "planned", "forecast"];

const PRESETS: { name: string; label: string; hint: string; d: [number, number, number] }[] = [
  { name: "baseline", label: "Baseline", hint: "no change", d: [0, 0, 0] },
  { name: "heatwave", label: "Heatwave", hint: "+6 C, +8% load, +2 h/day", d: [6, 0.08, 2] },
  { name: "mild_week", label: "Mild week", hint: "-4 C, -8% load, -1.5 h/day", d: [-4, -0.08, -1.5] },
  { name: "peak_occupancy", label: "Peak occupancy", hint: "+15% load, +3 h/day", d: [0, 0.15, 3] },
  { name: "night_setback", label: "Night setback", hint: "-20% load, -4 h/day", d: [0, -0.2, -4] },
];

/** A labelled slider that always shows its own units and its zero point. */
function Slider({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="flex items-baseline justify-between text-[11px]">
        <span style={{ color: "var(--ink-2)" }}>{label}</span>
        <span className="num font-semibold" style={{ color: value === 0 ? "var(--ink-2)" : "var(--ink)" }}>
          {format(value)}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
        aria-label={label}
      />
    </label>
  );
}

/** Horizon mix as a single stacked bar - the headline the panel exists for. */
function BucketBar({ counts, width = 150 }: { counts: Record<string, number>; width?: number }) {
  const total = BUCKETS.reduce((s, b) => s + (counts[b] ?? 0), 0) || 1;
  return (
    <div className="flex items-center gap-2">
      <div className="flex h-4 overflow-hidden rounded" style={{ width }}>
        {BUCKETS.map((b) => {
          const n = counts[b] ?? 0;
          if (!n) return null;
          return (
            <div
              key={b}
              title={`${n} ${BUCKET_LABEL[b]}`}
              style={{ width: `${(n / total) * 100}%`, background: bucketVar(b) }}
            />
          );
        })}
      </div>
      <span className="num text-[11px] whitespace-nowrap" style={{ color: "var(--ink-2)" }}>
        {BUCKETS.map((b) => counts[b] ?? 0).join(" / ")}
      </span>
    </div>
  );
}

/** before -> after, with the arrow only drawn when something actually moved. */
function Delta({
  before,
  after,
  dp = 1,
  unit = "",
  goodWhenLower = true,
}: {
  before: number;
  after: number;
  dp?: number;
  unit?: string;
  goodWhenLower?: boolean;
}) {
  const moved = Math.abs(after - before) > Math.pow(10, -dp) / 2;
  if (!moved) {
    return (
      <span className="num" style={{ color: "var(--ink-2)" }}>
        {num(before, dp)}
        {unit}
      </span>
    );
  }
  // Worse is marked with the urgent hue, which is the same meaning it carries
  // everywhere else in the app: this needs attention sooner.
  const worse = goodWhenLower ? after > before : after < before;
  return (
    <span className="num">
      <span style={{ color: "var(--ink-2)" }}>{num(before, dp)}</span>
      <span aria-hidden style={{ color: "var(--muted)" }}>
        {" → "}
      </span>
      <span className="font-semibold" style={{ color: worse ? "var(--urgent)" : "var(--ink)" }}>
        {num(after, dp)}
        {unit}
      </span>
    </span>
  );
}

export default function WhatIfPanel({ assetId }: { assetId: string }) {
  const [dAmb, setDAmb] = useState(0);
  const [dLoad, setDLoad] = useState(0);
  const [dDuty, setDDuty] = useState(0);
  const [data, setData] = useState<WhatIfResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const seq = useRef(0);

  // Debounced, and last-write-wins. Dragging a slider fires a request per
  // step otherwise, and out-of-order replies would make the panel flicker
  // between scenarios.
  const run = useCallback(
    (amb: number, load: number, duty: number) => {
      const mine = ++seq.current;
      setBusy(true);
      api
        .whatif({ asset_id: assetId, d_ambient_c: amb, d_load_frac: load, d_duty_hours: duty })
        .then((r) => {
          if (mine !== seq.current) return;
          setData(r);
          setError(null);
        })
        .catch((e) => {
          if (mine === seq.current) setError(e);
        })
        .finally(() => {
          if (mine === seq.current) setBusy(false);
        });
    },
    [assetId],
  );

  useEffect(() => {
    const t = setTimeout(() => run(dAmb, dLoad, dDuty), 180);
    return () => clearTimeout(t);
  }, [dAmb, dLoad, dDuty, run]);

  // A new asset starts from its own baseline rather than inheriting a slider
  // position set on the previous one.
  useEffect(() => {
    setDAmb(0);
    setDLoad(0);
    setDDuty(0);
  }, [assetId]);

  const applyPreset = (d: [number, number, number]) => {
    setDAmb(d[0]);
    setDLoad(d[1]);
    setDDuty(d[2]);
  };

  const active = PRESETS.find((p) => p.d[0] === dAmb && p.d[1] === dLoad && p.d[2] === dDuty);
  const movedParts = (data?.parts_delta ?? []).filter((p) => p.delta !== 0);
  const bBy = new Map((data?.components_before ?? []).map((c) => [c.component_id, c]));
  const moved = (data?.components_after ?? [])
    .map((c) => ({ after: c, before: bBy.get(c.component_id) }))
    .filter((r) => r.before && r.before.bucket !== r.after.bucket);

  return (
    <Card
      title="What if conditions change?"
      subtitle="The sliders move operating conditions only. The twin, the TTM bands and the parts call are all recomputed from them - nothing here sets a prediction directly."
      right={
        <div className="flex items-center gap-1.5">
          {data && (
            <Pill title="Server-side compute for this scenario">{num(data.compute_ms, 0)} ms</Pill>
          )}
          {busy && (
            <span className="text-[11px]" style={{ color: "var(--muted)" }}>
              recomputing...
            </span>
          )}
        </div>
      }
    >
      <div className="flex flex-col gap-3">
        {/* ---- presets ---- */}
        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              type="button"
              title={p.hint}
              onClick={() => applyPreset(p.d)}
              className="rounded px-2 py-1 text-[11px]"
              style={{
                border: "1px solid var(--line)",
                background: active?.name === p.name ? "var(--plane)" : "transparent",
                fontWeight: active?.name === p.name ? 600 : 400,
              }}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* ---- sliders ---- */}
        <div className="grid gap-3 sm:grid-cols-3">
          <Slider
            label="Ambient"
            value={dAmb}
            min={-10}
            max={12}
            step={0.5}
            format={(v) => `${v > 0 ? "+" : ""}${num(v, 1)} C`}
            onChange={setDAmb}
          />
          <Slider
            label="Load"
            value={dLoad}
            min={-0.3}
            max={0.3}
            step={0.01}
            format={(v) => `${v > 0 ? "+" : ""}${pct(v, 0)}`}
            onChange={setDLoad}
          />
          <Slider
            label="Running hours"
            value={dDuty}
            min={-8}
            max={8}
            step={0.5}
            format={(v) => `${v > 0 ? "+" : ""}${num(v, 1)} h/day`}
            onChange={setDDuty}
          />
        </div>

        {error ? (
          <p className="rounded px-2.5 py-2 text-[11px]" style={{ background: "var(--urgent-wash)" }}>
            What-if is unavailable: {String((error as Error)?.message ?? error)}. The rest of the
            screen is unaffected - every other number is precomputed.
          </p>
        ) : !data ? (
          <p className="text-[11px]" style={{ color: "var(--ink-2)" }}>
            Computing the baseline...
          </p>
        ) : (
          <>
            {/* ---- headline: the horizon mix ---- */}
            <div
              className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded px-2.5 py-2"
              style={{ background: "var(--plane)" }}
            >
              <div className="flex flex-col gap-1">
                <span className="text-[10px]" style={{ color: "var(--ink-2)" }}>
                  Now
                </span>
                <BucketBar counts={data.bucket_counts_before} />
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-[10px]" style={{ color: "var(--ink-2)" }}>
                  {data.scenario_label}
                </span>
                <BucketBar counts={data.bucket_counts_after} />
              </div>
              {Object.keys(data.bucket_migration).length > 0 ? (
                <ul className="flex flex-col gap-0.5 text-[11px]">
                  {Object.entries(data.bucket_migration).map(([k, n]) => (
                    <li key={k}>
                      <strong className="num">{n}</strong> {k.includes("urgent") ? "" : ""}
                      {k.replace("->", "→")}
                    </li>
                  ))}
                </ul>
              ) : (
                <span className="text-[11px]" style={{ color: "var(--ink-2)" }}>
                  No component crosses a horizon boundary under this scenario.
                </span>
              )}
            </div>

            {/* ---- twin state ---- */}
            <div>
              <h4 className="mb-1 text-[11px] font-semibold">
                Healthy baseline from the physics twin
                <span className="ml-1 font-normal" style={{ color: "var(--ink-2)" }}>
                  - what a sound machine of this design would read in these conditions
                </span>
              </h4>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] sm:grid-cols-3">
                {(
                  [
                    ["Condensing", "t_cond_c", 1, " C", true],
                    ["Head pressure", "discharge_p_kpa", 0, " kPa", true],
                    ["Suction", "suction_p_kpa", 0, " kPa", false],
                    ["Plant COP", "cop", 2, "", false],
                    ["Capacity", "capacity_kw", 1, " kW", false],
                    ["Pressure ratio", "pressure_ratio", 2, "", true],
                  ] as [string, keyof typeof data.twin_before, number, string, boolean][]
                ).map(([label, key, dp, unit, lowerIsBetter]) => (
                  <div key={label} className="flex items-baseline justify-between gap-2">
                    <dt style={{ color: "var(--ink-2)" }}>{label}</dt>
                    <dd>
                      <Delta
                        before={data.twin_before[key] as number}
                        after={data.twin_after[key] as number}
                        dp={dp}
                        unit={unit}
                        goodWhenLower={lowerIsBetter}
                      />
                    </dd>
                  </div>
                ))}
              </dl>
              {(data.twin_after.cycling || data.twin_after.head_pressure_limited) && (
                <p className="mt-1 text-[11px]" style={{ color: "var(--ink-2)" }}>
                  {data.twin_after.cycling && (
                    <>
                      Below the unloading floor the machine cycles rather than modulating (duty{" "}
                      <span className="num">{pct(data.twin_after.duty_frac, 0)}</span>), so the gauges
                      hold still while only the efficiency moves.{" "}
                    </>
                  )}
                  {data.twin_after.head_pressure_limited &&
                    "Minimum-condensing control is holding the head pressure up."}
                </p>
              )}
            </div>

            {/* ---- what moved ---- */}
            {moved.length > 0 && (
              <div>
                <h4 className="mb-1 text-[11px] font-semibold">
                  Components that change horizon ({moved.length})
                </h4>
                <table className="w-full text-[11px]">
                  <thead className="text-left text-[10px]" style={{ color: "var(--muted)" }}>
                    <tr>
                      <th className="py-1 font-normal">Component</th>
                      <th className="py-1 font-normal">Horizon</th>
                      <th className="py-1 text-right font-normal">P50 TTM</th>
                      <th className="py-1 text-right font-normal">P(fail 14 d)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {moved.map(({ before, after }) => (
                      <tr key={after.component_id} style={{ borderTop: "1px solid var(--line)" }}>
                        <td className="py-1">{titleCase(after.component_type)}</td>
                        <td className="py-1">
                          <span className="inline-flex items-center gap-1">
                            <BucketChip bucket={before!.bucket} />
                            <span aria-hidden style={{ color: "var(--muted)" }}>
                              {"→"}
                            </span>
                            <BucketChip bucket={after.bucket} />
                          </span>
                        </td>
                        <td className="num py-1 text-right">
                          {days(before!.p50_days)}
                          <span aria-hidden style={{ color: "var(--muted)" }}>
                            {" → "}
                          </span>
                          <span
                            className="font-semibold"
                            style={{
                              color:
                                after.p50_days < before!.p50_days ? "var(--urgent)" : "var(--ink)",
                            }}
                          >
                            {days(after.p50_days)}
                          </span>
                        </td>
                        <td className="num py-1 text-right">
                          <Delta
                            before={before!.p_fail_14d * 100}
                            after={after.p_fail_14d * 100}
                            dp={1}
                            unit="%"
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* ---- parts ---- */}
            {movedParts.length > 0 && (
              <div>
                <h4 className="mb-1 text-[11px] font-semibold">
                  Parts this asset would need in the next 14 days
                  <span className="ml-1 font-normal" style={{ color: "var(--ink-2)" }}>
                    - P90 of a Monte Carlo over this asset alone
                  </span>
                </h4>
                <ul className="flex flex-wrap gap-1.5">
                  {movedParts.map((p) => (
                    <li
                      key={p.part_no}
                      className="num rounded px-2 py-1 text-[11px]"
                      style={{
                        border: "1px solid var(--line)",
                        background: p.delta > 0 ? bucketWash("urgent") : "transparent",
                      }}
                    >
                      <span className="font-medium">{p.part_no}</span>{" "}
                      <span style={{ color: "var(--ink-2)" }}>
                        {num(p.p90_before, 1)}
                        {" → "}
                      </span>
                      <span className="font-semibold">{num(p.p90_after, 1)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* ---- assumptions, always visible rather than behind a toggle ---- */}
            <details>
              <summary className="cursor-pointer text-[11px]" style={{ color: "var(--ink-2)" }}>
                What this scenario assumes ({data.assumptions.length})
              </summary>
              <ul className="mt-1 flex flex-col gap-1 text-[11px]" style={{ color: "var(--ink-2)" }}>
                {data.assumptions.map((a) => (
                  <li key={a} className="flex gap-1.5">
                    <span aria-hidden>&middot;</span>
                    <span>{a}</span>
                  </li>
                ))}
              </ul>
            </details>
          </>
        )}
      </div>
    </Card>
  );
}
