// Screen 1 - Command Center. The entry point for the demo click path:
// a red tile leads to a hub, a hub leads to an urgent job, an urgent job leads
// to the Asset Twin. Every tile and every hub is a filter on the whole screen.

import { BandBar, BandLegend, HubMap } from "../components/charts";
import {
  Card,
  ConfidenceBadge,
  ErrorNote,
  Pill,
  Spinner,
  StatTile,
} from "../components/ui";
import { api } from "../lib/api";
import { days, money, num, pct } from "../lib/fmt";
import { useAsync } from "../lib/useAsync";

export default function CommandCenter({
  hub,
  setHub,
  onOpenAsset,
  onGoto,
}: {
  hub: string | null;
  setHub: (h: string | null) => void;
  onOpenAsset: (assetId: string) => void;
  onGoto: (screen: "parts" | "planner") => void;
}) {
  const { data, error, loading } = useAsync(() => api.kpis(hub ?? undefined), [hub]);

  if (error) return <ErrorNote error={error} />;
  if (!data) return <Spinner label="Loading fleet state" />;

  const fmtTile = (key: string, v: number) =>
    key === "excess_value" ? money(v) : num(v, 0);

  return (
    <div className={loading ? "opacity-60 transition-opacity" : "transition-opacity"}>
      {/* Filters sit in one row above everything, per the interaction spec */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Pill title="Date the pipeline was last built">as of {data.as_of_date}</Pill>
        <Pill title="Hours since the newest telemetry reading">
          telemetry {num(data.data_freshness_hours, 0)} h old
        </Pill>
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
          <Pill>all 8 hubs</Pill>
        )}
      </div>

      <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {data.tiles.map((t) => (
          <StatTile
            key={t.key}
            label={t.label}
            value={fmtTile(t.key, t.value)}
            unit={t.unit}
            tone={t.tone}
            onClick={() => {
              if (t.key === "projected_stockouts" || t.key === "excess_value") onGoto("planner");
              else if (t.key === "expedites_avoided") onGoto("planner");
            }}
          />
        ))}
      </div>

      <div className="grid gap-3 lg:grid-cols-[1.05fr_1fr]">
        <Card
          title="Hubs by urgent load"
          subtitle="Click a hub to filter every screen to it"
          right={
            <Pill>
              {data.totals.recommendations} recommendations - {data.totals.transfers} transfers
            </Pill>
          }
        >
          <HubMap hubs={data.hubs} selected={hub} onSelect={setHub} />
        </Card>

        <Card
          title="Urgent job queue"
          subtitle={`${data.urgent_queue.length} components with a repair predicted inside 14 days`}
          bodyClass="p-0"
        >
          <div className="max-h-[330px] overflow-y-auto">
            <table className="w-full text-[12px]">
              <thead
                className="sticky top-0 text-left text-[10px]"
                style={{ background: "var(--surface)", color: "var(--muted)" }}
              >
                <tr>
                  <th className="px-3 py-2 font-normal">Asset / component</th>
                  <th className="px-2 py-2 font-normal">TTM band</th>
                  <th className="px-2 py-2 text-right font-normal">P(fail&nbsp;14d)</th>
                  <th className="px-2 py-2 font-normal">Confidence</th>
                  <th className="px-3 py-2 font-normal">Parts</th>
                </tr>
              </thead>
              <tbody>
                {data.urgent_queue.map((j) => (
                  <tr
                    key={j.component_id}
                    className="cursor-pointer hover:brightness-[0.98]"
                    style={{ borderTop: "1px solid var(--line)" }}
                    onClick={() => onOpenAsset(j.asset_id)}
                  >
                    <td className="px-3 py-2">
                      <div className="font-medium">{j.asset_label}</div>
                      <div className="text-[11px]" style={{ color: "var(--ink-2)" }}>
                        {j.component_type.replace(/_/g, " ")} &middot; {j.site}
                      </div>
                    </td>
                    <td className="px-2 py-2">
                      <BandBar
                        p10={j.p10_days}
                        p50={j.p50_days}
                        p90={j.p90_days}
                        bucket="urgent"
                        width={120}
                      />
                      <div className="num text-[10px]" style={{ color: "var(--ink-2)" }}>
                        {days(j.p10_days)} - {days(j.p90_days)}
                      </div>
                    </td>
                    <td className="num px-2 py-2 text-right font-medium">{pct(j.p_fail_14d)}</td>
                    <td className="px-2 py-2">
                      <ConfidenceBadge level={j.confidence} reasons={j.confidence_reasons} />
                    </td>
                    <td className="px-3 py-2">
                      {j.parts_short ? (
                        <span
                          className="rounded px-1.5 py-0.5 text-[11px] font-medium"
                          style={{ background: "var(--urgent-wash)", color: "var(--ink)" }}
                        >
                          short
                        </span>
                      ) : (
                        <span className="text-[11px]" style={{ color: "var(--ink-2)" }}>
                          on hand
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-3 py-2" style={{ borderTop: "1px solid var(--line)" }}>
            <BandLegend />
          </div>
        </Card>
      </div>

      <p className="mt-3 text-[11px]" style={{ color: "var(--muted)" }}>
        Every number here is read from the precomputed pipeline - no model runs in the request
        path. Uncertainty is shown as a band, never a single figure.
      </p>
    </div>
  );
}
