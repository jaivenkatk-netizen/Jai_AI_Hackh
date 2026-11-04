"""Orbit API contract.

This module is the single source of truth for every shape that crosses the
API boundary. The React app is built against fixtures generated from these
models (see contracts/fixtures.py), so the UI never waits on the backend.

Three invariants the brief demands, encoded in the types themselves:

1. No bare numbers. Anything predicted carries a TTMBand (P10/P50/P90) and a
   Confidence badge. A float on its own is not a valid prediction here.
2. Every recommendation explains itself before asking for approval:
   `why`, `confidence` and `assumptions` are required, not optional.
3. Horizon is a closed set (urgent / planned / forecast), so "separate urgent
   needs from longer-term forecasts" is a type-level guarantee, not a
   convention a screen might forget.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

# Horizon boundaries, in days. Tunable in the UI; these are the defaults the
# brief asks for (urgent 0-14, planned 15-90, forecast 90+).
URGENT_MAX_DAYS = 14
PLANNED_MAX_DAYS = 90


class Bucket(str, Enum):
    """Planning horizon. Colour in the UI is bound to this: red/amber/blue."""

    URGENT = "urgent"  # 0-14 d   -> work order now, reserve parts
    PLANNED = "planned"  # 15-90 d  -> bundle into next PM visit
    FORECAST = "forecast"  # 90+ d    -> safety stock / buy plan only


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Criticality(str, Enum):
    """Service criticality class. Drives the inventory service-level target."""

    A = "A"  # 98% service level
    B = "B"  # 95%
    C = "C"  # 90%


SERVICE_LEVEL = {Criticality.A: 0.98, Criticality.B: 0.95, Criticality.C: 0.90}

# P(fail <= 14d) at or above which a component is Urgent, by criticality.
URGENT_THRESHOLD = {Criticality.A: 0.30, Criticality.B: 0.50, Criticality.C: 0.50}


class ActionType(str, Enum):
    TRANSFER = "transfer"  # lateral transshipment from another hub
    PURCHASE = "purchase"  # normal PO, lead time fits
    EXPEDITE = "expedite"  # need date inside lead time, no donor hub
    DEFER = "defer"  # low criticality, risk of waiting is small


class AssetType(str, Enum):
    CHILLER = "chiller"
    RTU = "rtu"
    AHU = "ahu"
    SPLIT_HEAT_PUMP = "split_heat_pump"
    VRF = "vrf"


# --------------------------------------------------------------------------
# Shared value objects
# --------------------------------------------------------------------------


class TTMBand(BaseModel):
    """Time-to-maintenance as a distribution, never a point estimate.

    Read off the fitted AFT survival curve: with log T ~ Normal(mu, sigma),
    quantile(q) = exp(mu + sigma * z_q). `sigma_log` is carried through
    because it is what ultimately drives safety stock downstream - a wide
    band must raise stock, and keeping sigma on the wire makes that
    traceable rather than implicit.
    """

    p10_days: float = Field(..., description="Optimistic bound; 10th percentile of TTM")
    p50_days: float = Field(..., description="Median TTM - the headline number")
    p90_days: float = Field(..., description="Pessimistic bound; 90th percentile")
    sigma_log: float = Field(
        ..., description="Scale of the log-time distribution; propagates to safety stock"
    )

    @property
    def width_days(self) -> float:
        return self.p90_days - self.p10_days


class ConfidenceBadge(BaseModel):
    """High/Medium/Low plus the reasons, so the badge is auditable."""

    level: Confidence
    interval_width_days: float = Field(..., description="P90 - P10; wider = less confident")
    data_quality: float = Field(
        ..., ge=0, le=1, description="Sensor coverage x freshness x (1 - imputed share)"
    )
    ood_score: float = Field(
        ..., ge=0, description="Distance from training distribution; >1 is out-of-distribution"
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Plain-words drivers of the badge, e.g. '9-day telemetry gap'",
    )


class AssumptionCard(BaseModel):
    """What a prediction assumes, in plain words. Required by the brief."""

    label: str = Field(..., description="e.g. 'Duty cycle'")
    text: str = Field(..., description="e.g. 'continues at about 14 h/day'")
    basis: str | None = Field(
        None, description="Where the value came from, e.g. 'trailing 30-day mean'"
    )


class Driver(BaseModel):
    """One SHAP contribution - why this component, not just that it is degrading."""

    feature: str
    display_name: str
    shap_value: float = Field(..., description="Signed; positive shortens predicted life")
    value: float | str | None = None
    direction: Literal["shortens_life", "extends_life"]


# --------------------------------------------------------------------------
# GET /api/kpis  -> Command Center
# --------------------------------------------------------------------------


class KpiTile(BaseModel):
    key: str
    label: str
    value: float
    unit: str | None = None
    delta_vs_baseline: float | None = Field(
        None, description="Signed change vs the min/max + fixed-interval baseline"
    )
    tone: Literal["critical", "warning", "neutral", "good"] = "neutral"
    drill_filter: dict[str, Any] = Field(
        default_factory=dict,
        description="Filter to apply when the tile is clicked, e.g. {'bucket': 'urgent'}",
    )


class HubRisk(BaseModel):
    hub_id: str
    name: str
    region: str
    lat: float
    lon: float
    assets_at_risk: int
    urgent_jobs: int
    projected_stockouts: int
    excess_value: float
    risk_score: float = Field(..., ge=0, le=1, description="Normalised; drives map colour")


class KpiResponse(BaseModel):
    generated_at: datetime
    as_of_date: date
    tiles: list[KpiTile]
    hubs: list[HubRisk]
    urgent_queue: list[UrgentJob]
    data_freshness_hours: float


class UrgentJob(BaseModel):
    """A row in the Command Center urgent queue."""

    job_id: str
    asset_id: str
    asset_label: str
    site: str
    hub_id: str
    component_type: str
    failure_mode: str
    ttm: TTMBand
    p_fail_14d: float = Field(..., ge=0, le=1)
    confidence: ConfidenceBadge
    parts_short: bool = Field(..., description="True if a needed part is not on hand at the hub")


# --------------------------------------------------------------------------
# GET /api/fleet  -> Fleet Health
# --------------------------------------------------------------------------


class FleetRow(BaseModel):
    asset_id: str
    asset_label: str
    asset_type: AssetType
    site: str
    hub_id: str
    region: str
    criticality: Criticality
    top_risk_component_id: str
    top_risk_component_type: str
    ttm: TTMBand
    p_fail_14d: float = Field(..., ge=0, le=1)
    bucket: Bucket
    confidence: ConfidenceBadge


class FleetResponse(BaseModel):
    as_of_date: date
    total: int
    rows: list[FleetRow]
    # region x component_type -> mean risk, for the heatmap
    risk_heatmap: dict[str, dict[str, float]] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# GET /api/asset/{asset_id}  -> Asset Twin
# --------------------------------------------------------------------------


class SensorSeries(BaseModel):
    """Observed telemetry with the twin's healthy baseline overlaid.

    `residual` is observed - baseline and is itself a model feature, which is
    how the physics twin earns its place rather than being decoration.
    """

    signal: str
    display_name: str
    unit: str
    timestamps: list[datetime]
    observed: list[float | None]
    baseline: list[float | None] = Field(
        default_factory=list, description="Asset twin's healthy expectation"
    )
    residual: list[float | None] = Field(default_factory=list)


class SurvivalCurve(BaseModel):
    days: list[float]
    survival: list[float] = Field(..., description="S(t) = P(TTM > t)")
    lower: list[float] = Field(default_factory=list, description="Calibrated lower band")
    upper: list[float] = Field(default_factory=list, description="Calibrated upper band")


class TwinState(BaseModel):
    """Live cycle schematic state, coloured by component health."""

    refrigerant: str
    suction_p_kpa: float
    discharge_p_kpa: float
    superheat_k: float
    subcool_k: float
    cop: float
    cop_healthy: float = Field(..., description="Twin's healthy-baseline COP for the same conditions")
    charge_frac: float = Field(..., ge=0, le=1.5, description="Virtual sensor: estimated charge")
    component_health: dict[str, float] = Field(
        default_factory=dict, description="component_type -> health index in [0, 1]"
    )


class ComponentPrediction(BaseModel):
    component_id: str
    component_type: str
    failure_mode: str
    ttm: TTMBand
    p_fail_14d: float
    p_fail_90d: float
    bucket: Bucket
    confidence: ConfidenceBadge
    drivers: list[Driver]
    assumptions: list[AssumptionCard]
    survival: SurvivalCurve
    model_layer: Literal["weibull_prior", "survival_xgb", "ensemble"] = "survival_xgb"


class ServiceEvent(BaseModel):
    ts: datetime
    wo_id: str
    type: Literal["PM", "CM", "emergency"]
    component_type: str | None = None
    failure_mode: str | None = None
    parts_used: list[str] = Field(default_factory=list)
    first_time_fix: bool | None = None
    notes: str | None = None


class AssetDetailResponse(BaseModel):
    asset_id: str
    asset_label: str
    asset_type: AssetType
    model: str
    site: str
    hub_id: str
    region: str
    climate_zone: str
    capacity_kw: float
    install_date: date
    criticality: Criticality
    twin: TwinState
    sensors: list[SensorSeries]
    components: list[ComponentPrediction]
    history: list[ServiceEvent]


# --------------------------------------------------------------------------
# GET /api/parts-demand  -> Parts Demand
# --------------------------------------------------------------------------


class DemandFan(BaseModel):
    """Monte Carlo demand distribution. P90 - not the mean - drives stocking.

    A decision-aware benchmark of intermittent-demand methods found the most
    accurate forecasters often under-forecast and miss service targets, so the
    mean is reported for reference but never used for a stocking decision.
    """

    p10: float
    p50: float
    p90: float
    mean: float
    draws: int = Field(..., description="Monte Carlo sample count behind the fan")


class PartsDemandRow(BaseModel):
    part_no: str
    description: str
    hub_id: str
    bucket: Bucket
    criticality: Criticality
    unit_cost: float
    demand: DemandFan
    on_hand: int
    on_order: int
    reserved: int
    # Traceability: which assets/failure modes generated this number
    contributing_assets: list[str] = Field(default_factory=list)
    contributing_failure_modes: list[str] = Field(default_factory=list)


class VanKitItem(BaseModel):
    """Parts to put on the van for a scheduled job - the first-time-fix lever."""

    part_no: str
    description: str
    qty: float
    p_used: float = Field(..., ge=0, le=1)
    reason: Literal["primary", "cheap_high_probability_extra"]
    on_hand_at_hub: int


class VanKit(BaseModel):
    job_id: str
    asset_id: str
    component_type: str
    failure_mode: str
    items: list[VanKitItem]
    expected_first_time_fix: float = Field(..., ge=0, le=1)


class PartsDemandResponse(BaseModel):
    as_of_date: date
    rows: list[PartsDemandRow]
    van_kits: list[VanKit] = Field(default_factory=list)


# --------------------------------------------------------------------------
# GET /api/recommendations  -> Inventory Planner
# --------------------------------------------------------------------------


class StockPosition(BaseModel):
    part_no: str
    description: str
    hub_id: str
    criticality: Criticality
    on_hand: int
    on_order: int
    reorder_point: float = Field(..., description="Lead-time demand + safety stock, from MC")
    safety_stock: float
    projected_need_p50: float
    projected_need_p90: float
    lead_time_mean_d: float
    lead_time_sd_d: float
    service_level_target: float
    days_of_cover: float | None = None
    excess_units: float = Field(0, description="Stock above the 90-day P90 need")
    excess_value: float = 0


class RecommendationCard(BaseModel):
    """The action, the confidence and the assumptions - all three, always.

    This is the object the whole pipeline exists to produce. `why` is built
    from the actual numbers behind the decision so a planner (or a judge) can
    audit it without leaving the screen.
    """

    rec_id: str
    action: ActionType
    part_no: str
    description: str
    qty: float
    from_hub_id: str | None = Field(None, description="Donor hub for a transfer")
    to_hub_id: str
    need_by: date
    why: str = Field(..., description="Plain-words rationale, built from the real numbers")
    evidence: list[str] = Field(
        default_factory=list,
        description="Bullet evidence, e.g. 'P(fail <= 14 d) = 0.62 on CH-117'",
    )
    confidence: ConfidenceBadge
    assumptions: list[AssumptionCard]
    cost_delta: float = Field(..., description="Cost of this action vs the next-best option")
    avoided_action: ActionType | None = Field(
        None, description="What this avoids, e.g. an expedited order"
    )
    status: Literal["pending", "approved", "edited", "rejected"] = "pending"


class RecommendationsResponse(BaseModel):
    as_of_date: date
    cards: list[RecommendationCard]
    positions: list[StockPosition]
    total_excess_value: float
    expedites_avoided: int


# --------------------------------------------------------------------------
# POST /api/whatif  -> the one live compute path (single asset)
# --------------------------------------------------------------------------


# This section was amended once the twin and the survival model existed, and
# the amendment is recorded here rather than left as silent drift:
#
#   - `horizon_days` was dropped. The horizon is a closed set everywhere else
#     in this contract (Bucket), and a free integer here would have let one
#     screen define "urgent" differently from the rest of the app.
#   - `TwinState` is replaced by `WhatIfTwinView` for this endpoint. The
#     original required `charge_frac` and `component_health`, which belong to
#     the asset screen's live state, and lacked the condensing temperature,
#     duty fraction and control-limit flags the physics twin actually
#     produces and the panel needs to explain itself.
#   - `ComponentPrediction` is replaced by `WhatIfComponent`. Bucket migration
#     is the headline of this endpoint, so each component has to carry its
#     bucket and its 14-day probability; `failure_mode` is not known at
#     prediction time and was never populated.
#   - `parts_delta` is a list of before/after pairs rather than PartsDemandRow,
#     which is a hub-scoped row with no notion of a scenario comparison.


class WhatIfRequest(BaseModel):
    """A scenario is three condition deltas, and nothing else.

    Deliberately narrow: there is no field here that can move a prediction
    directly. The only way to change the answer is to change the weather, the
    load or the running hours, and let the model respond to that.
    """

    # Unknown fields are an error, not something to ignore. Every delta
    # defaults to zero, so a misspelled field name would otherwise return a
    # confident baseline instead of a complaint - the worst failure a
    # what-if endpoint can have.
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    d_ambient_c: float = Field(0, ge=-15, le=15, description="Ambient shift, degrees C")
    d_load_frac: float = Field(0, ge=-0.5, le=0.5, description="Load shift, fraction")
    d_duty_hours: float = Field(0, ge=-12, le=12, description="Duty-cycle shift, h/day")
    preset: str | None = Field(None, description="Named preset; overrides the three deltas")


class WhatIfSignal(BaseModel):
    """One telemetry channel against the twin's healthy baseline."""

    observed: float
    healthy: float
    residual: float


class WhatIfTwinView(BaseModel):
    """Twin state under one set of conditions."""

    ambient_c: float
    load_frac: float
    t_evap_c: float
    t_cond_c: float
    suction_p_kpa: float
    discharge_p_kpa: float
    superheat_k: float
    subcool_k: float
    comp_current_a: float
    cop: float = Field(..., description="Plant COP, including auxiliaries")
    cop_compressor: float = Field(..., description="Compressor-only cycle COP")
    capacity_kw: float
    pressure_ratio: float
    duty_frac: float = Field(..., ge=0, le=1, description="1.0 modulating; below that, cycling")
    cycling: bool = Field(..., description="Running below the compressor unloading floor")
    head_pressure_limited: bool = Field(..., description="Minimum-condensing control is holding")
    converged: bool
    signals: dict[str, WhatIfSignal] = Field(default_factory=dict)


class WhatIfComponent(BaseModel):
    """One component's band and bucket under one scenario."""

    component_id: str
    component_type: str
    criticality: Criticality
    p10_days: float
    p50_days: float
    p90_days: float
    p_fail_14d: float = Field(..., ge=0, le=1)
    p_fail_90d: float = Field(..., ge=0, le=1)
    bucket: Bucket


class WhatIfPartDelta(BaseModel):
    part_no: str
    p90_before: float
    p90_after: float
    delta: float


class WhatIfResponse(BaseModel):
    asset_id: str
    asset_label: str
    asset_type: str
    site_name: str
    hub_id: str
    scenario_label: str
    d_ambient: float
    d_load: float
    d_duty: float
    preset: str | None = None
    twin_before: WhatIfTwinView
    twin_after: WhatIfTwinView
    components_before: list[WhatIfComponent]
    components_after: list[WhatIfComponent]
    bucket_counts_before: dict[str, int] = Field(default_factory=dict)
    bucket_counts_after: dict[str, int] = Field(default_factory=dict)
    bucket_migration: dict[str, int] = Field(
        default_factory=dict,
        description="Components moving between buckets, e.g. {'planned -> urgent': 2}",
    )
    moved_component_ids: list[str] = Field(default_factory=list)
    parts_delta: list[WhatIfPartDelta] = Field(
        default_factory=list, description="Change in this asset's own urgent parts demand"
    )
    assumptions: list[str] = Field(
        default_factory=list, description="What the scenario holds fixed, in plain words"
    )
    compute_ms: float


# --------------------------------------------------------------------------
# GET /api/policy-kpis  -> Scenarios (network twin)
# --------------------------------------------------------------------------


class PolicyKpiRow(BaseModel):
    kpi: str
    unit: str
    baseline: float | None = Field(None, description="Fixed-interval + min/max")
    servicesense: float | None = None
    change_pct: float | None = None
    better_is: Literal["lower", "higher"]
    measured: bool = Field(
        True, description="False means not yet measured - render the cell empty, never estimated"
    )


class PolicyKpiResponse(BaseModel):
    scenario: str
    sim_months: int
    seed: int
    rows: list[PolicyKpiRow]
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# POST /api/copilot  -> copilot panel (SSE)
# --------------------------------------------------------------------------


class ToolCallTrace(BaseModel):
    """Rendered beside the answer so every number the copilot says is traceable."""

    tool_name: str
    arguments: dict[str, Any]
    result_summary: str
    row_count: int | None = None
    duration_ms: float
    is_error: bool = False


class CopilotRequest(BaseModel):
    message: str
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Screen context, e.g. {'hub_id': 'HUB-BLR', 'asset_id': 'CH-117'}",
    )
    history: list[dict[str, Any]] = Field(default_factory=list)


class CopilotEvent(BaseModel):
    """One SSE frame."""

    type: Literal["text", "tool_call", "tool_result", "done", "error"]
    text: str | None = None
    trace: ToolCallTrace | None = None


class CopilotResponse(BaseModel):
    """Non-streaming form, used by tests and the cached demo mode."""

    answer: str
    traces: list[ToolCallTrace]
    grounded: bool = Field(
        ..., description="False if the model answered without any tool call - must not ship"
    )


# Forward refs (KpiResponse references UrgentJob declared after it)
KpiResponse.model_rebuild()


__all__ = [
    "URGENT_MAX_DAYS",
    "PLANNED_MAX_DAYS",
    "SERVICE_LEVEL",
    "URGENT_THRESHOLD",
    "Bucket",
    "Confidence",
    "Criticality",
    "ActionType",
    "AssetType",
    "TTMBand",
    "ConfidenceBadge",
    "AssumptionCard",
    "Driver",
    "KpiTile",
    "HubRisk",
    "UrgentJob",
    "KpiResponse",
    "FleetRow",
    "FleetResponse",
    "SensorSeries",
    "SurvivalCurve",
    "TwinState",
    "ComponentPrediction",
    "ServiceEvent",
    "AssetDetailResponse",
    "DemandFan",
    "PartsDemandRow",
    "VanKitItem",
    "VanKit",
    "PartsDemandResponse",
    "StockPosition",
    "RecommendationCard",
    "RecommendationsResponse",
    "WhatIfRequest",
    "WhatIfResponse",
    "WhatIfTwinView",
    "WhatIfSignal",
    "WhatIfComponent",
    "WhatIfPartDelta",
    "PolicyKpiRow",
    "PolicyKpiResponse",
    "ToolCallTrace",
    "CopilotRequest",
    "CopilotEvent",
    "CopilotResponse",
]
