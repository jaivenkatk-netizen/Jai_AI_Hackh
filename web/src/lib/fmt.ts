// Formatting. Centralised so a TTM never renders as a bare precise number in
// one place and a capped one in another.

export const MAX_TTM_DAYS = 3650;

/** Days, with the planning-horizon cap made explicit rather than implied. */
export function days(v: number | null | undefined, beyond = false): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  if (beyond || v >= MAX_TTM_DAYS) return "> 10 yr";
  if (v < 1) return "< 1 d";
  if (v < 100) return `${v.toFixed(v < 10 ? 1 : 0)} d`;
  if (v < 730) return `${Math.round(v)} d`;
  return `${(v / 365).toFixed(1)} yr`;
}

export const pct = (v: number | null | undefined, dp = 0) =>
  v === null || v === undefined || Number.isNaN(v) ? "-" : `${(v * 100).toFixed(dp)}%`;

export const num = (v: number | null | undefined, dp = 0) =>
  v === null || v === undefined || Number.isNaN(v)
    ? "-"
    : v.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });

/** Indian-market money, since the synthetic fleet is priced in INR. */
export function money(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  const a = Math.abs(v);
  if (a >= 1e7) return `${(v / 1e7).toFixed(2)} Cr`;
  if (a >= 1e5) return `${(v / 1e5).toFixed(2)} L`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return v.toFixed(0);
}

export const titleCase = (s: string) =>
  s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

export const shortDate = (iso: string) =>
  new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short" });

export const ACTION_LABEL: Record<string, string> = {
  transfer: "Transfer",
  purchase: "Purchase order",
  expedite: "Expedite",
  defer: "Defer / reschedule",
};
