// Physics-informed fault state: what the inverse network infers about this
// machine's physical condition from its sensors.
//
// The reason this panel looks the way it does: of the three parameters the
// network infers, only one survived validation. Showing all three as equal
// readings would be the dishonest option, and dropping the other two would
// hide that the method was tried and partly failed. So each parameter carries
// its validation state, the unvalidated ones are visibly demoted, and the
// physics residual - how well the inferred state actually reproduces the
// sensors - is shown rather than buried.

import type { FaultState } from "../lib/api";
import { num, pct } from "../lib/fmt";
import { Card, Pill } from "./ui";

/** Horizontal gauge. Charge reads as a level; degradation reads as an extent. */
function Gauge({
  value,
  lo,
  hi,
  invert,
  muted,
}: {
  value: number;
  lo: number;
  hi: number;
  invert: boolean;
  muted: boolean;
}) {
  const frac = Math.min(Math.max((value - lo) / (hi - lo), 0), 1);
  // "Worse" is to the left for charge (less refrigerant) and to the right for
  // degradation, so the filled portion always means the healthy part.
  const fill = invert ? frac : 1 - frac;
  return (
    <div
      className="relative h-2 w-full overflow-hidden rounded"
      style={{ background: "var(--grid)" }}
    >
      <div
        className="h-full rounded"
        style={{
          width: `${fill * 100}%`,
          background: muted ? "var(--grid)" : "var(--forecast)",
          opacity: muted ? 1 : 0.9,
          borderRight: muted ? "none" : "2px solid var(--forecast)",
        }}
      />
    </div>
  );
}

export default function FaultStatePanel({ state }: { state: FaultState | null }) {
  if (!state) return null;

  const validated = state.params.filter((p) => p.validated);
  const unvalidated = state.params.filter((p) => !p.validated);

  return (
    <Card
      title="Inferred physical condition"
      subtitle="A network inverted through the twin's physics. No labels were used to train it - the governing equations supply the training signal, because nothing in the field carries a 'the condenser is 23% fouled' label."
      right={
        <Pill title="Date of the telemetry this was inferred from">as of {state.as_of}</Pill>
      }
    >
      <div className="flex flex-col gap-3">
        {validated.map((p) => (
          <div key={p.key}>
            <div className="flex items-baseline justify-between gap-2 text-[11px]">
              <span className="font-medium">{p.label}</span>
              <span className="num font-semibold">
                {p.key === "charge_frac" ? pct(p.value, 0) : pct(p.value, 0)}
              </span>
            </div>
            <div className="mt-1">
              <Gauge
                value={p.value}
                lo={p.key === "charge_frac" ? 0.65 : 0}
                hi={p.key === "charge_frac" ? 1.05 : 0.6}
                invert={p.key === "charge_frac"}
                muted={false}
              />
            </div>
            <p className="mt-1 text-[10px]" style={{ color: "var(--ink-2)" }}>
              {p.key === "charge_frac"
                ? "Validated at r = 0.79 against known refrigerant-circuit damage. Superheat is the signal that carries charge, and it is the one channel the twin reproduces to within a tenth of the sensor noise."
                : "Validated against known damage."}
            </p>
          </div>
        ))}

        {unvalidated.length > 0 && (
          <div
            className="rounded px-2.5 py-2"
            style={{ background: "var(--plane)", border: "1px dashed var(--line)" }}
          >
            <h5 className="text-[11px] font-semibold">
              Inferred but not validated - shown for completeness, not for decisions
            </h5>
            <ul className="mt-1.5 flex flex-col gap-1.5">
              {unvalidated.map((p) => (
                <li key={p.key}>
                  <div className="flex items-baseline justify-between gap-2 text-[11px]">
                    <span style={{ color: "var(--ink-2)" }}>{p.label}</span>
                    <span className="num" style={{ color: "var(--muted)" }}>
                      {pct(p.value, 0)}
                    </span>
                  </div>
                  <div className="mt-1">
                    <Gauge value={p.value} lo={0} hi={0.6} invert={false} muted />
                  </div>
                </li>
              ))}
            </ul>
            <p className="mt-1.5 text-[10px]" style={{ color: "var(--ink-2)" }}>
              These two track ambient temperature (r = -0.78) more than the damage they are
              meant to measure (r = 0.23), and most assets sit pinned against the parameter
              bound. The cause is the same one recorded for the twin residual: this fleet's
              pressure response to weather is not thermodynamically consistent, so the pressure
              channels cannot be inverted. Superheat escapes that, which is why charge works.
            </p>
          </div>
        )}

        <div className="text-[10px]" style={{ color: "var(--muted)" }}>
          Physics residual{" "}
          <span className="num">{num(state.physics_residual, 0)}</span> squared noise sigmas
          {state.explains_sensors
            ? " - the inferred state reproduces this machine's sensors to within the noise."
            : " - well above the 1.0 a sound forward model would give, so the inferred state does"
              + " not fully explain this machine's sensors. Reported rather than hidden."}
        </div>
      </div>
    </Card>
  );
}
