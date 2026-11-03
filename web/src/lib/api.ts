// API client. One base URL, so flipping between fixtures and the live backend
// is a single flag rather than a rewrite of every fetch call.

export const USE_FIXTURES = import.meta.env.VITE_USE_FIXTURES === "1";
const BASE = "/api";

export type Bucket = "urgent" | "planned" | "forecast";
export type Confidence = "high" | "medium" | "low";
export type ActionType = "transfer" | "purchase" | "expedite" | "defer";

export interface AssumptionCard {
  label: string;
  text: string;
  basis?: string | null;
}

export interface Driver {
  feature: string;
  display_name: string;
  shap_value: number;
  value: number | null;
  direction: "shortens_life" | "extends_life";
  rank: number;
}

export interface KpiTile {
  key: string;
  label: string;
  value: number;
  unit?: string | null;
  tone: "critical" | "warning" | "neutral" | "good";
  drill_filter: Record<string, unknown>;
}

export interface HubRisk {
  hub_id: string;
  name: string;
  region: string;
  lat: number;
  lon: number;
  assets_at_risk: number;
  urgent_jobs: number;
  projected_stockouts: number;
  excess_value: number;
  risk_score: number;
}

export interface UrgentJob {
  component_id: string;
  asset_id: string;
  asset_label: string;
  site: string;
  hub_id: string;
  component_type: string;
  p10_days: number;
  p50_days: number;
  p90_days: number;
  p_fail_14d: number;
  confidence: Confidence;
  confidence_reasons: string[];
  assumptions: AssumptionCard[];
  parts_short: boolean;
}

export interface KpiResponse {
  generated_at: string;
  as_of_date: string;
  tiles: KpiTile[];
  hubs: HubRisk[];
  urgent_queue: UrgentJob[];
  data_freshness_hours: number;
  totals: { recommendations: number; transfers: number };
}

export interface FleetRow {
  asset_id: string;
  asset_label: string;
  asset_type: string;
  site: string;
  hub_id: string;
  region: string;
  criticality: "A" | "B" | "C";
  top_risk_component_type: string;
  p10_days: number;
  p50_days: number;
  p90_days: number;
  beyond_horizon: boolean;
  p_fail_14d: number;
  p_fail_90d: number;
  bucket: Bucket;
  confidence: Confidence;
  confidence_reasons: string[];
  interval_width_days: number;
  data_quality: number;
}

export interface FleetResponse {
  as_of_date: string;
  total: number;
  rows: FleetRow[];
  risk_heatmap: Record<string, Record<string, number>>;
}

export interface ComponentPrediction {
  component_id: string;
  component_type: string;
  p10_days: number;
  p50_days: number;
  p90_days: number;
  beyond_horizon: boolean;
  sigma_effective: number;
  mu_log_days: number;
  p_fail_14d: number;
  p_fail_30d: number;
  p_fail_90d: number;
  bucket: Bucket;
  confidence: Confidence;
  confidence_reasons: string[];
  assumptions: AssumptionCard[];
  drivers: Driver[];
  survival: { days: number[]; survival: number[] };
  data_quality: number;
  interval_width_days: number;
}

export interface SensorSeries {
  signal: string;
  display_name: string;
  unit: string;
  timestamps: string[];
  observed: (number | null)[];
  baseline: (number | null)[];
  baseline_source: string;
  residual: (number | null)[];
}

export interface AssetDetail {
  asset_id: string;
  asset_label: string;
  asset_type: string;
  model: string;
  refrigerant: string;
  site_name: string;
  hub_id: string;
  region: string;
  climate_zone: string;
  capacity_kw: number;
  install_date: string;
  criticality: "A" | "B" | "C";
  twin: {
    refrigerant: string | null;
    suction_p_kpa: number | null;
    discharge_p_kpa: number | null;
    superheat_k: number | null;
    cop: number | null;
    ambient_t_max: number | null;
    component_health: Record<string, number>;
    source: string;
  };
  sensors: SensorSeries[];
  components: ComponentPrediction[];
  history: {
    wo_id: string;
    ts: string;
    type: string;
    component_type: string | null;
    failure_mode: string | null;
    parts_used: string[];
    first_time_fix: boolean | null;
    notes: string | null;
    downtime_h: number | null;
  }[];
  fault_state: FaultState | null;
}

export interface PartsDemandRow {
  part_no: string;
  description: string;
  hub_id: string;
  bucket: Bucket;
  criticality: "A" | "B" | "C";
  unit_cost: number;
  demand_p10: number;
  demand_p50: number;
  demand_p90: number;
  demand_mean: number;
  pm_demand: number;
  total_p50: number;
  total_p90: number;
  draws: number;
  on_hand: number | null;
  available: number | null;
  reorder_point: number | null;
  safety_stock: number | null;
  shortfall_vs_p90: number | null;
  contributing: {
    asset_id: string;
    component_id: string;
    component_type: string;
    failure_mode: string;
    p_contribution: number;
  }[];
  contributing_assets: string[];
  contributing_failure_modes: string[];
}

export interface VanKitRow {
  component_type: string;
  failure_mode: string;
  size_class: string;
  refrigerant: string;
  part_no: string;
  description: string;
  qty: number;
  p_used: number;
  reason: string;
  unit_cost: number;
  expected_first_time_fix: number | null;
}

export interface PartsDemandResponse {
  as_of_date: string;
  rows: PartsDemandRow[];
  van_kits: VanKitRow[];
}

export interface RecommendationCard {
  rec_id: string;
  action: ActionType;
  part_no: string;
  description: string;
  qty: number;
  from_hub_id: string | null;
  to_hub_id: string;
  need_by: string;
  need_in_days: number;
  criticality: "A" | "B" | "C";
  why: string;
  evidence: string[];
  confidence: Confidence;
  assumptions: AssumptionCard[];
  cost_delta: number;
  action_cost: number;
  avoided_action: ActionType | null;
  unit_cost: number;
  status: string;
}

export interface StockPosition {
  part_no: string;
  description: string;
  hub_id: string;
  criticality: "A" | "B" | "C";
  on_hand: number;
  on_order: number;
  available: number;
  reorder_point: number;
  safety_stock: number;
  projected_need_p50: number;
  projected_need_p90: number;
  shortfall_vs_p90: number;
  excess_units: number;
  excess_value: number;
  lead_time_mean_d: number;
  lead_time_sd_d: number;
  service_level_target: number;
  days_of_cover: number | null;
}

export interface RecommendationsResponse {
  as_of_date: string;
  cards: RecommendationCard[];
  positions: StockPosition[];
  total_excess_value: number;
  expedites_avoided: number;
}

export interface MetaResponse {
  as_of_date: string;
  counts: Record<string, number>;
  eval_table: Record<string, unknown>[];
  eval_by_type: Record<string, unknown>[];
  calibration_curve: Record<string, unknown>[];
  coverage_by_type: Record<string, unknown>[];
  model: Record<string, unknown>;
  data_sources: { name: string; role: string; detail: string }[];
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} on ${path}`);
  return (await res.json()) as T;
}

const qs = (p: Record<string, string | number | undefined | null>) => {
  const s = new URLSearchParams();
  for (const [k, v] of Object.entries(p)) if (v !== undefined && v !== null && v !== "") s.set(k, String(v));
  const out = s.toString();
  return out ? `?${out}` : "";
};

// ---------------------------------------------------------------------------
// What-if. The only endpoint that runs the model inside a request, so it is
// the only one that POSTs and the only one with a perceptible latency budget.
// ---------------------------------------------------------------------------

export interface FaultParam {
  key: string;
  label: string;
  value: number;
  /** False means the parameter did not survive validation and must be shown as such. */
  validated: boolean;
}

export interface FaultState {
  params: FaultParam[];
  physics_residual: number;
  explains_sensors: boolean;
  as_of: string;
  method: string;
}

export interface TwinSignal {
  observed: number;
  healthy: number;
  residual: number;
}

export interface TwinView {
  ambient_c: number;
  load_frac: number;
  t_evap_c: number;
  t_cond_c: number;
  suction_p_kpa: number;
  discharge_p_kpa: number;
  superheat_k: number;
  subcool_k: number;
  comp_current_a: number;
  cop: number;
  cop_compressor: number;
  capacity_kw: number;
  pressure_ratio: number;
  duty_frac: number;
  cycling: boolean;
  head_pressure_limited: boolean;
  converged: boolean;
  signals: Record<string, TwinSignal>;
}

export interface WhatIfComponent {
  component_id: string;
  component_type: string;
  criticality: "A" | "B" | "C";
  p10_days: number;
  p50_days: number;
  p90_days: number;
  p_fail_14d: number;
  p_fail_90d: number;
  bucket: Bucket;
}

export interface WhatIfPartDelta {
  part_no: string;
  p90_before: number;
  p90_after: number;
  delta: number;
}

export interface WhatIfResponse {
  asset_id: string;
  asset_label: string;
  asset_type: string;
  site_name: string;
  hub_id: string;
  scenario_label: string;
  d_ambient: number;
  d_load: number;
  d_duty: number;
  preset?: string;
  twin_before: TwinView;
  twin_after: TwinView;
  components_before: WhatIfComponent[];
  components_after: WhatIfComponent[];
  bucket_counts_before: Record<Bucket, number>;
  bucket_counts_after: Record<Bucket, number>;
  bucket_migration: Record<string, number>;
  moved_component_ids: string[];
  parts_delta: WhatIfPartDelta[];
  assumptions: string[];
  compute_ms: number;
}

export interface WhatIfScenario {
  asset_id: string;
  d_ambient_c?: number;
  d_load_frac?: number;
  d_duty_hours?: number;
  preset?: string;
}

const post = async <T>(path: string, body: unknown): Promise<T> => {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} on ${path}`);
  return (await r.json()) as T;
};

export const api = {
  whatif: (s: WhatIfScenario) => post<WhatIfResponse>("/whatif", s),
  kpis: (hub?: string) => get<KpiResponse>(`/kpis${qs({ hub })}`),
  fleet: (hub?: string, bucket?: Bucket, limit = 300) =>
    get<FleetResponse>(`/fleet${qs({ hub, bucket, limit })}`),
  asset: (id: string) => get<AssetDetail>(`/asset/${encodeURIComponent(id)}`),
  partsDemand: (hub?: string, bucket?: Bucket, limit = 400) =>
    get<PartsDemandResponse>(`/parts-demand${qs({ hub, bucket, limit })}`),
  recommendations: (hub?: string, action?: string, limit = 200) =>
    get<RecommendationsResponse>(`/recommendations${qs({ hub, action, limit })}`),
  hubs: () => get<HubRisk[]>("/hubs"),
  meta: () => get<MetaResponse>("/meta"),
};
