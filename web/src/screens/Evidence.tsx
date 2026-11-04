// Screen 5 - Real-world Evidence. The synthetic fleet supplies the HVAC physics
// and the parts/stock links; this screen shows the method checked against three
// public datasets we did not generate, and says plainly where it falls short.

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, ErrorNote, Pill, Spinner } from "../components/ui";
import type { EvidenceResponse } from "../lib/api";
import { api } from "../lib/api";
import { num } from "../lib/fmt";
import { useAsync } from "../lib/useAsync";

const AXIS = { fontSize: 11, fill: "var(--muted)" };
const SIGNAL_LABEL: Record<string, string> = {
  "kW/Ton": "Efficiency (kW/ton)",
  TRC_sub: "Liquid subcooling",
  Tsh_suc: "Suction superheat",
  Tsh_dis: "Discharge superheat",
  PRC: "Condenser pressure",
  PRE: "Evaporator pressure",
  TCA: "Condenser approach",
  TEA: "Evaporator approach",
};

function zTone(z: number | null): { bg: string; fg: string } {
  // |z| < 3 is inside normal scatter: leave it uncoloured so only real shifts stand out.
  if (z == null || Math.abs(z) < 3) return { bg: "transparent", fg: "var(--ink-2)" };
  const a = Math.min(Math.abs(z) / 10, 1);
  const hue = z > 0 ? "urgent" : "planned";
  return { bg: `color-mix(in srgb, var(--${hue}) ${20 + a * 50}%, transparent)`, fg: "var(--ink)" };
}

function Ashrae({ d }: { d: NonNullable<EvidenceResponse["ashrae"]> }) {
  return (
    <Card
      title="ASHRAE RP-1043 - which signal catches which chiller fault"
      subtitle={`${d.runs} lab runs, ${d.steady_states} steady states. Each cell is the shift from the fault-free baseline at the worst severity tested, in baseline standard deviations. Shaded = beyond 3 SD.`}
      right={
        <Pill title="fault-free kW/Ton scatter after the load/temperature baseline">
          baseline sd {num(d.baseline_sd_pct, 1)}%
        </Pill>
      }
    >
      <div className="overflow-x-auto">
        <table className="w-full text-[12px]">
          <thead>
            <tr style={{ color: "var(--ink-2)" }}>
              <th className="py-1 pr-3 text-left font-medium">Fault (max severity)</th>
              {d.signals.map((s) => (
                <th key={s} className="px-2 py-1 text-right font-medium">
                  {SIGNAL_LABEL[s] ?? s}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {d.faults.map((f) => {
              const top = f.levels[f.levels.length - 1];
              return (
                <tr key={f.fault} style={{ borderTop: "1px solid var(--line)" }}>
                  <td className="py-1 pr-3">
                    {f.fault} <span style={{ color: "var(--muted)" }}>({top.severity})</span>
                  </td>
                  {d.signals.map((s) => {
                    const z = top.z[s];
                    const t = zTone(z);
                    return (
                      <td
                        key={s}
                        className="num px-2 py-1 text-right"
                        style={{ background: t.bg, color: t.fg }}
                      >
                        {z == null ? "-" : (z > 0 ? "+" : "") + num(z, 1)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[11px]" style={{ color: "var(--ink-2)" }}>
        Read honestly: refrigerant-side signals (subcooling, condenser pressure and approach)
        separate leaks, overcharge and non-condensables sharply. Evaporator flow and excess oil
        barely move any of them, so the fault-state network cannot see those faults from these
        sensors.
      </p>
    </Card>
  );
}

function Lbnl({ d }: { d: NonNullable<EvidenceResponse["lbnl"]> }) {
  const data = d.scenarios
    .filter((s) => s.energy_penalty_pct != null)
    .slice(0, 18)
    .map((s) => ({
      name: s.scenario.replace(/_/g, " ") + (s.comparable ? "" : " *"),
      v: s.energy_penalty_pct as number,
      ok: s.comparable,
    }));
  return (
    <Card
      title="LBNL chiller plant - energy cost of each plant fault"
      subtitle={`Plant power per unit cooling load on days with real demand, vs the fault-free year (${num(d.baseline_plant_kw, 0)} kW mean). ${d.scenarios.length} fault scenarios, one simulated year each.`}
    >
      <div style={{ height: Math.max(220, data.length * 22) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 130, right: 24 }}>
            <CartesianGrid stroke="var(--grid)" horizontal={false} />
            <XAxis type="number" tick={AXIS} unit="%" />
            <YAxis type="category" dataKey="name" tick={AXIS} width={125} />
            <Tooltip formatter={(v) => `${num(Number(v), 2)} %`} />
            <ReferenceLine x={0} stroke="var(--ink-2)" />
            <Bar dataKey="v" name="Energy penalty">
              {data.map((r) => (
                <Cell key={r.name} fill={r.v > 0 ? "var(--urgent)" : "var(--planned)"} fillOpacity={r.ok ? 1 : 0.35} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-[11px]" style={{ color: "var(--ink-2)" }}>
        * Faded bars: that simulated year served a different total cooling load than the fault-free
        year (bypass faults and some sensor-bias runs), so the % is not like-for-like. A negative
        number is a fault that saves energy while degrading service, e.g. a low chilled-water
        pressure reading.
      </p>
    </Card>
  );
}

function Ims({ d }: { d: NonNullable<EvidenceResponse["ims"]> }) {
  const tests = d.summary.map((s) => s.test);
  const [test, setTest] = useState(tests[0] ?? 1);
  const s = d.summary.find((x) => x.test === test);
  const series = d.series
    .filter((x) => x.test === test)
    .map((x) => ({ ...x, days: x.hours / 24 }));
  const thr = series[0]?.alarm_threshold;
  return (
    <Card
      title="NASA IMS bearings - run to failure"
      subtitle="Real remaining-life curves. Health indicator = vibration RMS of the bearing that failed; alarm = 5 SD above its healthy baseline for 3 consecutive snapshots."
      right={
        <div className="flex gap-1">
          {tests.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTest(t)}
              className="rounded px-2 py-0.5 text-[12px]"
              style={{
                background: t === test ? "var(--plane)" : "transparent",
                fontWeight: t === test ? 600 : 400,
              }}
            >
              Test {t}
            </button>
          ))}
        </div>
      }
    >
      <div style={{ height: 240 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={series} margin={{ left: 8, right: 24 }}>
            <CartesianGrid stroke="var(--grid)" />
            <XAxis
              dataKey="days"
              type="number"
              tick={AXIS}
              domain={[0, "dataMax"]}
              tickFormatter={(v) => `${Math.round(v)} d`}
            />
            <YAxis tick={AXIS} domain={["auto", "auto"]} />
            <Tooltip
              formatter={(v) => num(Number(v), 3)}
              labelFormatter={(v) => `day ${num(Number(v), 1)}`}
            />
            {thr != null && (
              <ReferenceLine
                y={thr}
                stroke="var(--urgent)"
                strokeDasharray="4 3"
                label={{ value: "alarm", fill: "var(--urgent)", fontSize: 11 }}
              />
            )}
            <Line
              type="monotone"
              dataKey="hi"
              name="Vibration RMS"
              dot={false}
              stroke="var(--ink)"
              strokeWidth={1.5}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {s && (
        <div className="mt-3 grid grid-cols-2 gap-3 text-[12px] md:grid-cols-4">
          <div>
            <div style={{ color: "var(--muted)" }}>Failure</div>
            {s.failure_mode}
          </div>
          <div>
            <div style={{ color: "var(--muted)" }}>Run length</div>
            <span className="num">{num(s.life_days, 1)} days</span>
          </div>
          <div>
            <div style={{ color: "var(--muted)" }}>Alarm lead time</div>
            <span className="num">
              {s.lead_time_days == null
                ? "no alarm"
                : `${num(s.lead_time_days, 1)} days (${num(s.lead_pct_of_life ?? 0, 0)}% of life)`}
            </span>
          </div>
          <div>
            <div style={{ color: "var(--muted)" }}>Rank corr. vs remaining life</div>
            <span className="num">
              {num(s.rank_corr_hi_vs_rul, 2)} all, {num(s.rank_corr_last_half, 2)} last half
            </span>
          </div>
        </div>
      )}
    </Card>
  );
}

export default function Evidence() {
  const { data, error, loading } = useAsync(() => api.evidence(), []);
  if (loading && !data) return <Spinner label="Loading evidence" />;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;
  return (
    <div className="flex flex-col gap-4">
      <p className="text-[12px]" style={{ color: "var(--ink-2)" }}>
        The synthetic fleet supplies HVAC physics, parts and stock. These three public datasets are
        independent of it, so they test whether the method holds on data we did not generate.
      </p>
      {data.ashrae ? (
        <Ashrae d={data.ashrae} />
      ) : (
        <Card title="ASHRAE RP-1043">Not loaded. Run python -m sim.real_data --source ashrae.</Card>
      )}
      {data.lbnl ? (
        <Lbnl d={data.lbnl} />
      ) : (
        <Card title="LBNL chiller plant">Not loaded. Run python -m sim.real_data --source lbnl.</Card>
      )}
      {data.ims ? (
        <Ims d={data.ims} />
      ) : (
        <Card title="NASA IMS bearings">Not loaded. Run python -m sim.real_data --source ims.</Card>
      )}
    </div>
  );
}
