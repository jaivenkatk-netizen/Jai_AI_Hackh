// Shared primitives. Hand-rolled rather than pulled from a component library:
// the set needed is small, and every piece here encodes one of the design
// rules, which is easier to guarantee in 20 lines than to configure.

import type { ReactNode } from "react";
import type { Bucket, Confidence } from "../lib/api";

export const BUCKET_LABEL: Record<Bucket, string> = {
  urgent: "Urgent",
  planned: "Planned",
  forecast: "Forecast",
};

export const BUCKET_RANGE: Record<Bucket, string> = {
  urgent: "0-14 d",
  planned: "15-90 d",
  forecast: "90+ d",
};

export const bucketVar = (b: Bucket) => `var(--${b})`;
export const bucketWash = (b: Bucket) => `var(--${b}-wash)`;

/** Horizon chip. The colour never travels without the word beside it. */
export function BucketChip({ bucket, range = false }: { bucket: Bucket; range?: boolean }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded px-1.5 py-0.5 text-[11px] font-medium whitespace-nowrap"
      style={{ background: bucketWash(bucket), color: "var(--ink)" }}
    >
      <span
        aria-hidden
        className="inline-block h-2 w-2 shrink-0 rounded-sm"
        style={{ background: bucketVar(bucket) }}
      />
      {BUCKET_LABEL[bucket]}
      {range && <span style={{ color: "var(--ink-2)" }}>{BUCKET_RANGE[bucket]}</span>}
    </span>
  );
}

const CONF_DOTS: Record<Confidence, number> = { high: 3, medium: 2, low: 1 };

/**
 * Confidence badge - dots plus a word, deliberately no hue.
 *
 * Using green/amber/red here would collide with the horizon scale, so a
 * low-confidence urgent row would read as a different kind of thing. Encoding
 * it as filled dots keeps colour meaning exactly one thing on every screen.
 */
export function ConfidenceBadge({
  level,
  reasons = [],
  compact = false,
}: {
  level: Confidence;
  reasons?: string[];
  compact?: boolean;
}) {
  const filled = CONF_DOTS[level] ?? 2;
  const title = reasons.length ? `${level} confidence - ${reasons.join("; ")}` : `${level} confidence`;
  return (
    <span
      className="inline-flex items-center gap-1.5 whitespace-nowrap text-[11px]"
      title={title}
      style={{ color: "var(--ink-2)" }}
    >
      <span aria-hidden className="inline-flex gap-[2px]">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="inline-block h-[7px] w-[7px] rounded-full"
            style={{
              background: i < filled ? "var(--ink)" : "transparent",
              border: `1px solid ${i < filled ? "var(--ink)" : "var(--axis)"}`,
            }}
          />
        ))}
      </span>
      {!compact && <span className="capitalize">{level}</span>}
      <span className="sr-only">{title}</span>
    </span>
  );
}

export function Card({
  title,
  subtitle,
  right,
  children,
  className = "",
  bodyClass = "p-4",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || right) && (
        <header
          className="flex items-start justify-between gap-3 px-4 pt-3 pb-2"
          style={{ borderBottom: "1px solid var(--line)" }}
        >
          <div>
            {title && <h2 className="text-[13px] font-semibold">{title}</h2>}
            {subtitle && (
              <p className="mt-0.5 text-[11px]" style={{ color: "var(--ink-2)" }}>
                {subtitle}
              </p>
            )}
          </div>
          {right}
        </header>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  );
}

/**
 * Stat tile. A single headline number is not a chart - per the form heuristic
 * it gets a tile, with the comparison as text rather than a one-bar plot.
 */
export function StatTile({
  label,
  value,
  unit,
  tone = "neutral",
  delta,
  onClick,
  active = false,
}: {
  label: string;
  value: string;
  unit?: string | null;
  tone?: "critical" | "warning" | "neutral" | "good";
  delta?: string | null;
  onClick?: () => void;
  active?: boolean;
}) {
  const accent =
    tone === "critical"
      ? "var(--urgent)"
      : tone === "warning"
        ? "var(--planned)"
        : tone === "good"
          ? "var(--good)"
          : "var(--forecast)";
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className="card w-full cursor-pointer px-4 py-3 text-left transition-shadow hover:shadow-sm"
      style={{ outline: active ? `2px solid ${accent}` : undefined }}
    >
      <div className="flex items-center gap-2">
        <span aria-hidden className="inline-block h-2 w-2 rounded-sm" style={{ background: accent }} />
        <span className="text-[11px]" style={{ color: "var(--ink-2)" }}>
          {label}
        </span>
      </div>
      <div className="mt-1.5 flex items-baseline gap-1.5">
        <span className="text-2xl leading-none font-semibold">{value}</span>
        {unit && (
          <span className="text-[11px]" style={{ color: "var(--muted)" }}>
            {unit}
          </span>
        )}
      </div>
      {delta && (
        <div className="mt-1 text-[11px]" style={{ color: "var(--ink-2)" }}>
          {delta}
        </div>
      )}
    </button>
  );
}

export function Pill({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-0.5 text-[11px] whitespace-nowrap"
      title={title}
      style={{ border: "1px solid var(--line)", color: "var(--ink-2)" }}
    >
      {children}
    </span>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 p-6 text-[12px]" style={{ color: "var(--ink-2)" }}>
      <span
        className="inline-block h-3 w-3 animate-spin rounded-full"
        style={{ border: "2px solid var(--axis)", borderTopColor: "var(--ink)" }}
      />
      {label}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div
      className="m-4 rounded p-3 text-[12px]"
      style={{ background: "var(--urgent-wash)", border: "1px solid var(--line)" }}
    >
      <strong>Could not load.</strong> {msg}
      <div className="mt-1" style={{ color: "var(--ink-2)" }}>
        Is the API running? <code>uvicorn api.main:app --port 8000</code>
      </div>
    </div>
  );
}

/** Toggle that reveals the underlying table - the relief rule for sub-3:1 fills. */
export function TableToggle({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="rounded px-2 py-1 text-[11px]"
      style={{ border: "1px solid var(--line)", color: "var(--ink-2)" }}
    >
      {open ? "Hide table" : "View as table"}
    </button>
  );
}
