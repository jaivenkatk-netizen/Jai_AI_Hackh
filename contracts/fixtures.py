"""Generate fixture JSON for the React app from the Pydantic contract.

Two jobs:

1. Unblock the front end. The UI is built against these files and flipped to
   the live API by changing one base URL.
2. Prove the contract round-trips. Every fixture is constructed as a real
   model instance, so a schema change that breaks the UI breaks this script
   first.

The fixture data is deliberately the *demo narrative* from the plan - chiller
CH-117 at Bengaluru with condenser fouling, P50 ~9 days, resolved by a
transfer of 2 compressors from Chennai rather than an expedited order. Building
the UI against this means the screens already tell the right story before any
model has been trained.

Run:  python -m contracts.fixtures
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

from contracts.rules import confidence_badge, p_fail_within, ttm_band
from contracts.schemas import (
    ActionType,
    AssetDetailResponse,
    AssetType,
    AssumptionCard,
    Bucket,
    ComponentPrediction,
    Confidence,
    ConfidenceBadge,
    CopilotResponse,
    Criticality,
    DemandFan,
    Driver,
    FleetResponse,
    FleetRow,
    HubRisk,
    KpiResponse,
    KpiTile,
    PartsDemandResponse,
    PartsDemandRow,
    PolicyKpiResponse,
    PolicyKpiRow,
    RecommendationCard,
    RecommendationsResponse,
    SensorSeries,
    ServiceEvent,
    StockPosition,
    SurvivalCurve,
    ToolCallTrace,
    TwinState,
    UrgentJob,
    VanKit,
    VanKitItem,
    WhatIfResponse,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "src" / "fixtures"
AS_OF = date(2026, 11, 3)
NOW = datetime(2026, 11, 3, 8, 30)

# The hero asset: a chiller degrading from condenser fouling.
# mu/sigma are in log-days, as the AFT model emits them.
HERO_MU, HERO_SIGMA = 2.20, 0.42  # exp(2.20) ~ 9.0 days
HERO_BAND = ttm_band(HERO_MU, HERO_SIGMA)
HERO_P14 = p_fail_within(HERO_MU, HERO_SIGMA, 14)

HUBS = [
    ("HUB-BLR", "Bengaluru", "South", 12.97, 77.59, 14, 3, 3, 18_400, 0.92),
    ("HUB-MAA", "Chennai", "South", 13.08, 80.27, 9, 1, 0, 61_200, 0.41),
    ("HUB-PNQ", "Pune", "West", 18.52, 73.86, 11, 2, 1, 27_500, 0.63),
    ("HUB-BOM", "Mumbai", "West", 19.08, 72.88, 7, 1, 0, 33_100, 0.35),
    ("HUB-DEL", "Delhi", "North", 28.70, 77.10, 12, 2, 1, 22_700, 0.58),
    ("HUB-HYD", "Hyderabad", "South", 17.39, 78.49, 6, 0, 0, 15_900, 0.24),
    ("HUB-AMD", "Ahmedabad", "West", 23.02, 72.57, 8, 1, 0, 19_300, 0.44),
    ("HUB-CCU", "Kolkata", "East", 22.57, 88.36, 5, 0, 0, 12_100, 0.19),
]


def _badge(level: Confidence, width: float, quality: float, reasons: list[str]) -> ConfidenceBadge:
    return ConfidenceBadge(
        level=level, interval_width_days=width, data_quality=quality, ood_score=0.4, reasons=reasons
    )


def _hero_badge() -> ConfidenceBadge:
    # Medium: a 9-day telemetry gap on two of three contributing assets.
    return confidence_badge(
        band=HERO_BAND,
        sensor_coverage=0.83,
        hours_since_last_reading=26.0,
        imputed_share=0.06,
        ood_score=0.4,
    )


def _survival(mu: float, sigma: float) -> SurvivalCurve:
    import math

    from scipy.stats import norm

    days = [float(d) for d in range(0, 121, 2)]
    surv, lo, hi = [], [], []
    for d in days:
        if d <= 0:
            surv.append(1.0), lo.append(1.0), hi.append(1.0)
            continue
        z = (math.log(d) - mu) / sigma
        s = float(1 - norm.cdf(z))
        surv.append(s)
        lo.append(float(max(0.0, 1 - norm.cdf((math.log(d) - mu) / (sigma * 1.25)))))
        hi.append(float(min(1.0, 1 - norm.cdf((math.log(d) - mu) / (sigma * 0.80)))))
    return SurvivalCurve(days=days, survival=surv, lower=lo, upper=hi)


def build_kpis() -> KpiResponse:
    tiles = [
        KpiTile(
            key="assets_at_risk",
            label="Assets at risk",
            value=72,
            unit="assets",
            tone="warning",
            drill_filter={"bucket": "urgent"},
        ),
        KpiTile(
            key="urgent_jobs",
            label="Urgent jobs (0-14 d)",
            value=10,
            unit="jobs",
            tone="critical",
            drill_filter={"bucket": "urgent"},
        ),
        KpiTile(
            key="projected_stockouts",
            label="Projected stockouts",
            value=3,
            unit="parts",
            delta_vs_baseline=-9,
            tone="critical",
            drill_filter={"shortage": True},
        ),
        KpiTile(
            key="expedites_avoided",
            label="Expedites avoided",
            value=7,
            unit="orders",
            delta_vs_baseline=7,
            tone="good",
            drill_filter={"action": "transfer"},
        ),
        KpiTile(
            key="excess_value",
            label="Excess stock value",
            value=210_200,
            unit="INR",
            delta_vs_baseline=-86_400,
            tone="neutral",
            drill_filter={"excess": True},
        ),
    ]
    hubs = [
        HubRisk(
            hub_id=h,
            name=n,
            region=r,
            lat=la,
            lon=lo,
            assets_at_risk=ar,
            urgent_jobs=uj,
            projected_stockouts=ps,
            excess_value=ev,
            risk_score=rs,
        )
        for (h, n, r, la, lo, ar, uj, ps, ev, rs) in HUBS
    ]
    queue = [
        UrgentJob(
            job_id="WO-1042",
            asset_id="CH-117",
            asset_label="Chiller CH-117",
            site="Prestige Tech Park",
            hub_id="HUB-BLR",
            component_type="compressor",
            failure_mode="mechanical_wear",
            ttm=HERO_BAND,
            p_fail_14d=HERO_P14,
            confidence=_hero_badge(),
            parts_short=True,
        ),
        UrgentJob(
            job_id="WO-1043",
            asset_id="RTU-204",
            asset_label="RTU-204",
            site="Embassy Golf Links",
            hub_id="HUB-BLR",
            component_type="condenser_fan_motor",
            failure_mode="bearing_failure",
            ttm=ttm_band(2.48, 0.38),
            p_fail_14d=p_fail_within(2.48, 0.38, 14),
            confidence=_badge(Confidence.HIGH, 9.1, 0.91, ["Narrow band, fresh telemetry"]),
            parts_short=False,
        ),
        UrgentJob(
            job_id="WO-1044",
            asset_id="CH-091",
            asset_label="Chiller CH-091",
            site="Manyata Embassy",
            hub_id="HUB-BLR",
            component_type="compressor",
            failure_mode="mechanical_wear",
            ttm=ttm_band(2.62, 0.47),
            p_fail_14d=p_fail_within(2.62, 0.47, 14),
            confidence=_badge(
                Confidence.MEDIUM, 17.4, 0.58, ["9-day telemetry gap", "6% of readings imputed"]
            ),
            parts_short=True,
        ),
    ]
    return KpiResponse(
        generated_at=NOW,
        as_of_date=AS_OF,
        tiles=tiles,
        hubs=hubs,
        urgent_queue=queue,
        data_freshness_hours=1.4,
    )


def build_fleet() -> FleetResponse:
    rows = [
        FleetRow(
            asset_id="CH-117",
            asset_label="Chiller CH-117",
            asset_type=AssetType.CHILLER,
            site="Prestige Tech Park",
            hub_id="HUB-BLR",
            region="South",
            criticality=Criticality.A,
            top_risk_component_id="CH-117-COMP-1",
            top_risk_component_type="compressor",
            ttm=HERO_BAND,
            p_fail_14d=HERO_P14,
            bucket=Bucket.URGENT,
            confidence=_hero_badge(),
        ),
        FleetRow(
            asset_id="RTU-204",
            asset_label="RTU-204",
            asset_type=AssetType.RTU,
            site="Embassy Golf Links",
            hub_id="HUB-BLR",
            region="South",
            criticality=Criticality.B,
            top_risk_component_id="RTU-204-FAN-1",
            top_risk_component_type="condenser_fan_motor",
            ttm=ttm_band(2.48, 0.38),
            p_fail_14d=p_fail_within(2.48, 0.38, 14),
            bucket=Bucket.URGENT,
            confidence=_badge(Confidence.HIGH, 9.1, 0.91, ["Narrow band, fresh telemetry"]),
        ),
        FleetRow(
            asset_id="VRF-312",
            asset_label="VRF-312",
            asset_type=AssetType.VRF,
            site="Cessna Business Park",
            hub_id="HUB-PNQ",
            region="West",
            criticality=Criticality.B,
            top_risk_component_id="VRF-312-EEV-2",
            top_risk_component_type="eev",
            ttm=ttm_band(3.71, 0.35),
            p_fail_14d=p_fail_within(3.71, 0.35, 14),
            bucket=Bucket.PLANNED,
            confidence=_badge(Confidence.HIGH, 23.0, 0.88, ["Narrow band, fresh telemetry"]),
        ),
        FleetRow(
            asset_id="AHU-058",
            asset_label="AHU-058",
            asset_type=AssetType.AHU,
            site="RMZ Infinity",
            hub_id="HUB-MAA",
            region="South",
            criticality=Criticality.C,
            top_risk_component_id="AHU-058-BELT-1",
            top_risk_component_type="belt",
            ttm=ttm_band(4.82, 0.30),
            p_fail_14d=p_fail_within(4.82, 0.30, 14),
            bucket=Bucket.FORECAST,
            confidence=_badge(
                Confidence.MEDIUM, 68.0, 0.52, ["No telemetry - fleet Weibull prior only"]
            ),
        ),
    ]
    return FleetResponse(
        as_of_date=AS_OF,
        total=500,
        rows=rows,
        risk_heatmap={
            "South": {"compressor": 0.71, "condenser_fan_motor": 0.44, "eev": 0.22, "belt": 0.12},
            "West": {"compressor": 0.38, "condenser_fan_motor": 0.51, "eev": 0.34, "belt": 0.15},
            "North": {"compressor": 0.29, "condenser_fan_motor": 0.33, "eev": 0.19, "belt": 0.11},
            "East": {"compressor": 0.18, "condenser_fan_motor": 0.21, "eev": 0.14, "belt": 0.09},
        },
    )


def build_asset_detail() -> AssetDetailResponse:
    ts = [NOW - timedelta(days=29 - i) for i in range(30)]
    # Condenser fouling: head pressure climbs away from the twin's baseline.
    disch_base = [2420.0 + 4.0 * i for i in range(30)]
    disch_obs = [2430.0 + 17.5 * i for i in range(30)]
    cop_base = [3.15 - 0.001 * i for i in range(30)]
    cop_obs = [3.12 - 0.0115 * i for i in range(30)]

    sensors = [
        SensorSeries(
            signal="discharge_p",
            display_name="Discharge pressure",
            unit="kPa",
            timestamps=ts,
            observed=disch_obs,
            baseline=disch_base,
            residual=[o - b for o, b in zip(disch_obs, disch_base)],
        ),
        SensorSeries(
            signal="cop",
            display_name="COP (virtual sensor)",
            unit="-",
            timestamps=ts,
            observed=cop_obs,
            baseline=cop_base,
            residual=[o - b for o, b in zip(cop_obs, cop_base)],
        ),
        SensorSeries(
            signal="comp_current",
            display_name="Compressor current",
            unit="A",
            timestamps=ts,
            observed=[41.2 + 0.21 * i for i in range(30)],
            baseline=[41.0 + 0.02 * i for i in range(30)],
            residual=[0.19 * i for i in range(30)],
        ),
    ]

    compressor = ComponentPrediction(
        component_id="CH-117-COMP-1",
        component_type="compressor",
        failure_mode="mechanical_wear",
        ttm=HERO_BAND,
        p_fail_14d=HERO_P14,
        p_fail_90d=p_fail_within(HERO_MU, HERO_SIGMA, 90),
        bucket=Bucket.URGENT,
        confidence=_hero_badge(),
        drivers=[
            Driver(
                feature="discharge_p_resid_30d_slope",
                display_name="Discharge pressure rising vs twin baseline",
                shap_value=0.83,
                value=17.5,
                direction="shortens_life",
            ),
            Driver(
                feature="cop_resid_30d",
                display_name="COP below healthy baseline",
                shap_value=0.61,
                value=-0.31,
                direction="shortens_life",
            ),
            Driver(
                feature="comp_current_30d_slope",
                display_name="Compressor current trending up",
                shap_value=0.44,
                value=0.21,
                direction="shortens_life",
            ),
            Driver(
                feature="short_cycle_count_7d",
                display_name="Short-cycling events this week",
                shap_value=0.27,
                value=19,
                direction="shortens_life",
            ),
            Driver(
                feature="days_since_pm",
                display_name="Recent PM visit",
                shap_value=-0.12,
                value=22,
                direction="extends_life",
            ),
        ],
        assumptions=[
            AssumptionCard(
                label="Duty cycle",
                text="continues at about 14 h/day",
                basis="trailing 30-day mean",
            ),
            AssumptionCard(
                label="Ambient",
                text="follows the 10-year average for November in climate zone 1A",
                basis="weather normal",
            ),
            AssumptionCard(
                label="Last PM",
                text="on 12 Oct reset the filter and belt clocks",
                basis="work order WO-0981",
            ),
            AssumptionCard(
                label="Data quality",
                text="6% of compressor-current readings imputed over a 26-hour gap",
                basis="telemetry completeness check",
            ),
        ],
        survival=_survival(HERO_MU, HERO_SIGMA),
        model_layer="survival_xgb",
    )

    fan = ComponentPrediction(
        component_id="CH-117-FAN-2",
        component_type="condenser_fan_motor",
        failure_mode="bearing_failure",
        ttm=ttm_band(3.95, 0.33),
        p_fail_14d=p_fail_within(3.95, 0.33, 14),
        p_fail_90d=p_fail_within(3.95, 0.33, 90),
        bucket=Bucket.PLANNED,
        confidence=_badge(Confidence.HIGH, 31.0, 0.89, ["Narrow band, fresh telemetry"]),
        drivers=[
            Driver(
                feature="vib_rms_30d",
                display_name="Vibration RMS elevated",
                shap_value=0.38,
                value=4.1,
                direction="shortens_life",
            )
        ],
        assumptions=[
            AssumptionCard(label="Duty cycle", text="continues at about 14 h/day", basis="trailing 30-day mean")
        ],
        survival=_survival(3.95, 0.33),
        model_layer="survival_xgb",
    )

    return AssetDetailResponse(
        asset_id="CH-117",
        asset_label="Chiller CH-117",
        asset_type=AssetType.CHILLER,
        model="CGAM-080-R32",
        site="Prestige Tech Park",
        hub_id="HUB-BLR",
        region="South",
        climate_zone="1A",
        capacity_kw=281.0,
        install_date=date(2019, 6, 14),
        criticality=Criticality.A,
        twin=TwinState(
            refrigerant="R32",
            suction_p_kpa=812.0,
            discharge_p_kpa=2945.0,
            superheat_k=6.2,
            subcool_k=3.1,
            cop=2.78,
            cop_healthy=3.12,
            charge_frac=0.97,
            component_health={
                "compressor": 0.41,
                "condenser": 0.38,
                "condenser_fan_motor": 0.74,
                "evaporator": 0.88,
                "eev": 0.92,
            },
        ),
        sensors=sensors,
        components=[compressor, fan],
        history=[
            ServiceEvent(
                ts=datetime(2026, 10, 12, 10, 15),
                wo_id="WO-0981",
                type="PM",
                parts_used=["FLT-24x24", "BELT-A42"],
                first_time_fix=True,
                notes="Quarterly PM. Condenser coil fouling noted, cleaned partially; recommend follow-up.",
            ),
            ServiceEvent(
                ts=datetime(2026, 7, 3, 14, 40),
                wo_id="WO-0774",
                type="CM",
                component_type="run_capacitor",
                failure_mode="capacitor_degradation",
                parts_used=["CAP-45-5"],
                first_time_fix=True,
                notes="Start current imbalance on compressor. Replaced run capacitor.",
            ),
        ],
    )


def build_parts_demand() -> PartsDemandResponse:
    rows = [
        PartsDemandRow(
            part_no="CMP-5T-R32",
            description="Compressor assembly, 5 ton, R32",
            hub_id="HUB-BLR",
            bucket=Bucket.URGENT,
            criticality=Criticality.A,
            unit_cost=94_500,
            demand=DemandFan(p10=1.0, p50=2.0, p90=3.0, mean=2.1, draws=1000),
            on_hand=1,
            on_order=0,
            reserved=1,
            contributing_assets=["CH-117", "CH-091", "CH-144"],
            contributing_failure_modes=["mechanical_wear", "burnout"],
        ),
        PartsDemandRow(
            part_no="FD-34-R32",
            description="Filter-drier, 3/4 in, R32",
            hub_id="HUB-BLR",
            bucket=Bucket.URGENT,
            criticality=Criticality.B,
            unit_cost=2_150,
            demand=DemandFan(p10=1.0, p50=2.0, p90=3.0, mean=2.0, draws=1000),
            on_hand=6,
            on_order=0,
            reserved=2,
            contributing_assets=["CH-117", "CH-091", "CH-144"],
            contributing_failure_modes=["mechanical_wear", "burnout"],
        ),
        PartsDemandRow(
            part_no="CNT-3P-40A",
            description="Contactor, 3-pole, 40 A",
            hub_id="HUB-PNQ",
            bucket=Bucket.URGENT,
            criticality=Criticality.B,
            unit_cost=1_480,
            demand=DemandFan(p10=4.0, p50=6.0, p90=9.0, mean=6.2, draws=1000),
            on_hand=2,
            on_order=0,
            reserved=0,
            contributing_assets=["RTU-118", "RTU-131", "VRF-312", "RTU-155", "AHU-077"],
            contributing_failure_modes=["contactor_pitting", "mechanical_wear"],
        ),
        PartsDemandRow(
            part_no="REF-R32-KG",
            description="Refrigerant R32 (kg)",
            hub_id="HUB-BLR",
            bucket=Bucket.PLANNED,
            criticality=Criticality.B,
            unit_cost=1_120,
            demand=DemandFan(p10=12.0, p50=22.5, p90=36.0, mean=23.4, draws=1000),
            on_hand=40,
            on_order=0,
            reserved=4,
            contributing_assets=["CH-117", "CH-091", "VRF-312"],
            contributing_failure_modes=["mechanical_wear", "refrigerant_leak"],
        ),
    ]
    van_kits = [
        VanKit(
            job_id="WO-1042",
            asset_id="CH-117",
            component_type="compressor",
            failure_mode="mechanical_wear",
            items=[
                VanKitItem(
                    part_no="CMP-5T-R32",
                    description="Compressor assembly, 5 ton, R32",
                    qty=1,
                    p_used=1.00,
                    reason="primary",
                    on_hand_at_hub=1,
                ),
                VanKitItem(
                    part_no="FD-34-R32",
                    description="Filter-drier, 3/4 in, R32",
                    qty=1,
                    p_used=0.95,
                    reason="primary",
                    on_hand_at_hub=6,
                ),
                VanKitItem(
                    part_no="REF-R32-KG",
                    description="Refrigerant R32 (kg)",
                    qty=4.5,
                    p_used=1.00,
                    reason="primary",
                    on_hand_at_hub=40,
                ),
                VanKitItem(
                    part_no="CNT-3P-40A",
                    description="Contactor, 3-pole, 40 A",
                    qty=1,
                    p_used=0.40,
                    reason="cheap_high_probability_extra",
                    on_hand_at_hub=5,
                ),
            ],
            expected_first_time_fix=0.93,
        )
    ]
    return PartsDemandResponse(as_of_date=AS_OF, rows=rows, van_kits=van_kits)


def build_recommendations() -> RecommendationsResponse:
    card = RecommendationCard(
        rec_id="REC-0001",
        action=ActionType.TRANSFER,
        part_no="CMP-5T-R32",
        description="Compressor assembly, 5 ton, R32",
        qty=2,
        from_hub_id="HUB-MAA",
        to_hub_id="HUB-BLR",
        need_by=date(2026, 11, 6),
        why=(
            "3 urgent jobs at Bengaluru sites need a compressor by 6 Nov; only 1 is on hand. "
            "Supplier lead time of 21 days is past the need date, so a normal PO cannot make it. "
            "Chennai holds 5 against a P90 need of 2, so transferring 2 covers Bengaluru without "
            "putting Chennai at risk - and avoids one expedited order."
        ),
        evidence=[
            "P(fail <= 14 d) = 0.62 on CH-117, 0.48 on CH-091, 0.41 on CH-144",
            "Bengaluru on hand: 1; P90 demand in 0-14 d: 3",
            "Chennai on hand: 5; Chennai P90 need over the same window: 2",
            "Supplier SUP-114 lead time 21 d (sd 4 d) > 3 d to need date",
            "Transfer time Chennai -> Bengaluru: 2 d",
        ],
        confidence=_hero_badge(),
        assumptions=[
            AssumptionCard(label="Transfer time", text="takes 2 days", basis="hub distance matrix"),
            AssumptionCard(
                label="Duty cycle", text="continues at current rate", basis="trailing 30-day mean"
            ),
            AssumptionCard(
                label="Data quality",
                text="two of three assets have a 9-day telemetry gap",
                basis="telemetry completeness check",
            ),
        ],
        cost_delta=-31_800,
        avoided_action=ActionType.EXPEDITE,
    )
    card2 = RecommendationCard(
        rec_id="REC-0002",
        action=ActionType.PURCHASE,
        part_no="CNT-3P-40A",
        description="Contactor, 3-pole, 40 A",
        qty=8,
        from_hub_id=None,
        to_hub_id="HUB-PNQ",
        need_by=date(2026, 11, 11),
        why=(
            "Pune's P90 contactor demand over the next 14 days is 9 against 2 on hand. "
            "Supplier lead time is 6 days, which fits before the 11 Nov need date, so a normal "
            "PO is the cheapest option that meets the date. No hub has surplus above its own P90."
        ),
        evidence=[
            "Pune P90 demand 0-14 d: 9 units; on hand 2",
            "Supplier SUP-032 lead time 6 d (sd 1.5 d) < 8 d to need date",
            "No donor hub above its P90 need",
            "Reorder point 7.4, safety stock 3.1 at 95% service level (class B)",
        ],
        confidence=_badge(
            Confidence.HIGH, 4.0, 0.9, ["Narrow band, fresh telemetry on all 5 contributing assets"]
        ),
        assumptions=[
            AssumptionCard(label="MOQ", text="supplier minimum order is 5 units", basis="parts master")
        ],
        cost_delta=11_840,
        avoided_action=None,
    )
    positions = [
        StockPosition(
            part_no="CMP-5T-R32",
            description="Compressor assembly, 5 ton, R32",
            hub_id="HUB-BLR",
            criticality=Criticality.A,
            on_hand=1,
            on_order=0,
            reorder_point=4.2,
            safety_stock=2.1,
            projected_need_p50=2.0,
            projected_need_p90=3.0,
            lead_time_mean_d=21.0,
            lead_time_sd_d=4.0,
            service_level_target=0.98,
            days_of_cover=4.5,
            excess_units=0,
            excess_value=0,
        ),
        StockPosition(
            part_no="CMP-5T-R32",
            description="Compressor assembly, 5 ton, R32",
            hub_id="HUB-MAA",
            criticality=Criticality.A,
            on_hand=5,
            on_order=0,
            reorder_point=3.1,
            safety_stock=1.6,
            projected_need_p50=1.0,
            projected_need_p90=2.0,
            lead_time_mean_d=21.0,
            lead_time_sd_d=4.0,
            service_level_target=0.98,
            days_of_cover=96.0,
            excess_units=3.0,
            excess_value=283_500,
        ),
    ]
    return RecommendationsResponse(
        as_of_date=AS_OF,
        cards=[card, card2],
        positions=positions,
        total_excess_value=210_200,
        expedites_avoided=7,
    )


def build_whatif() -> WhatIfResponse:
    detail = build_asset_detail()
    before = detail.components
    # Heatwave +6 C: higher ambient raises head pressure and stress, so TTM shortens.
    hot_mu, hot_sigma = 1.86, 0.44  # exp(1.86) ~ 6.4 days
    after_comp = before[0].model_copy(
        update={
            "ttm": ttm_band(hot_mu, hot_sigma),
            "p_fail_14d": p_fail_within(hot_mu, hot_sigma, 14),
            "p_fail_90d": p_fail_within(hot_mu, hot_sigma, 90),
            "survival": _survival(hot_mu, hot_sigma),
        }
    )
    hot_fan_mu, hot_fan_sigma = 3.62, 0.34
    after_fan = before[1].model_copy(
        update={
            "ttm": ttm_band(hot_fan_mu, hot_fan_sigma),
            "p_fail_14d": p_fail_within(hot_fan_mu, hot_fan_sigma, 14),
            "p_fail_90d": p_fail_within(hot_fan_mu, hot_fan_sigma, 90),
            "bucket": Bucket.URGENT,
            "survival": _survival(hot_fan_mu, hot_fan_sigma),
        }
    )
    twin_after = detail.twin.model_copy(
        update={
            "discharge_p_kpa": 3218.0,
            "cop": 2.49,
            "cop_healthy": 2.88,
            "component_health": {**detail.twin.component_health, "compressor": 0.33},
        }
    )
    return WhatIfResponse(
        asset_id="CH-117",
        scenario_label="Heatwave: ambient +6 degrees C for 2 weeks",
        twin_before=detail.twin,
        twin_after=twin_after,
        components_before=before,
        components_after=[after_comp, after_fan],
        parts_delta=[
            PartsDemandRow(
                part_no="CMP-5T-R32",
                description="Compressor assembly, 5 ton, R32",
                hub_id="HUB-BLR",
                bucket=Bucket.URGENT,
                criticality=Criticality.A,
                unit_cost=94_500,
                demand=DemandFan(p10=1.0, p50=2.0, p90=4.0, mean=2.4, draws=1000),
                on_hand=1,
                on_order=0,
                reserved=1,
                contributing_assets=["CH-117"],
                contributing_failure_modes=["mechanical_wear"],
            )
        ],
        bucket_migration={"planned->urgent": 1},
        compute_ms=412.0,
    )


def build_policy_kpis() -> PolicyKpiResponse:
    # Values left unmeasured stay None with measured=False - the UI renders an
    # empty cell. Never estimate a number into the judges' table.
    rows = [
        PolicyKpiRow(kpi="TTM MAE", unit="days", baseline=31.4, servicesense=11.8, change_pct=-62.4, better_is="lower"),
        PolicyKpiRow(kpi="P10-P90 coverage", unit="fraction", baseline=None, servicesense=0.81, change_pct=None, better_is="higher"),
        PolicyKpiRow(kpi="Stockouts per 100 jobs", unit="count", baseline=14.2, servicesense=4.1, change_pct=-71.1, better_is="lower"),
        PolicyKpiRow(kpi="Expedited orders", unit="count/yr", baseline=38.0, servicesense=11.0, change_pct=-71.1, better_is="lower"),
        PolicyKpiRow(kpi="Excess inventory value", unit="INR", baseline=1_040_000, servicesense=694_000, change_pct=-33.3, better_is="lower"),
        PolicyKpiRow(kpi="First-time fix rate", unit="fraction", baseline=0.68, servicesense=0.86, change_pct=26.5, better_is="higher"),
        PolicyKpiRow(kpi="Unplanned downtime", unit="hours/yr", baseline=612.0, servicesense=228.0, change_pct=-62.7, better_is="lower"),
    ]
    return PolicyKpiResponse(
        scenario="baseline_year",
        sim_months=12,
        seed=42,
        rows=rows,
        notes=[
            "PLACEHOLDER VALUES - replaced by sim/network_twin.py output before the demo.",
            "Held-out synthetic months 19-24, generator seed 1337 (different stress mix from training).",
        ],
    )


def build_copilot() -> CopilotResponse:
    return CopilotResponse(
        answer=(
            "Pune needs 6 contactors next week because five assets have a contactor-involving "
            "repair predicted inside 14 days: RTU-118 (P(fail) 0.58), RTU-131 (0.52), VRF-312 "
            "(0.44), RTU-155 (0.39) and AHU-077 (0.31). The compressor and fan-motor repair kits "
            "each carry a contactor at P(used) 0.40, which is what turns those five jobs into a "
            "P50 of 6 units and a P90 of 9. Pune holds 2 on hand, so the reorder point of 7.4 is "
            "breached. Supplier lead time is 6 days against an 11 Nov need date, so a normal PO "
            "covers it - no expedite needed."
        ),
        traces=[
            ToolCallTrace(
                tool_name="get_parts_demand",
                arguments={"hub_id": "HUB-PNQ", "part_no": "CNT-3P-40A", "bucket": "urgent"},
                result_summary="P50 6.0, P90 9.0, on_hand 2, reorder_point 7.4",
                row_count=1,
                duration_ms=18.2,
            ),
            ToolCallTrace(
                tool_name="get_fleet_risk",
                arguments={"hub_id": "HUB-PNQ", "bucket": "urgent"},
                result_summary="5 assets with contactor-involving repair inside 14 d",
                row_count=5,
                duration_ms=24.7,
            ),
            ToolCallTrace(
                tool_name="get_recommendations",
                arguments={"hub_id": "HUB-PNQ", "part_no": "CNT-3P-40A"},
                result_summary="REC-0002 purchase 8 units, lead time 6 d fits need date",
                row_count=1,
                duration_ms=11.4,
            ),
        ],
        grounded=True,
    )


FIXTURES: dict[str, BaseModel] = {}


def build_all() -> dict[str, BaseModel]:
    return {
        "kpis": build_kpis(),
        "fleet": build_fleet(),
        "asset_CH-117": build_asset_detail(),
        "parts_demand": build_parts_demand(),
        "recommendations": build_recommendations(),
        "whatif": build_whatif(),
        "policy_kpis": build_policy_kpis(),
        "copilot": build_copilot(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = build_all()
    for name, model in fixtures.items():
        path = OUT_DIR / f"{name}.json"
        path.write_text(
            json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  {path.relative_to(OUT_DIR.parents[2])}  ({path.stat().st_size:,} bytes)")
    index = OUT_DIR / "index.ts"
    lines = ["// Generated by contracts/fixtures.py - do not edit by hand.", ""]
    for name in fixtures:
        ident = name.replace("-", "_")
        lines.append(f'import {ident} from "./{name}.json";')
    lines.append("")
    lines.append("export const fixtures = {")
    for name in fixtures:
        ident = name.replace("-", "_")
        lines.append(f'  "{name}": {ident},')
    lines.append("};")
    lines.append("")
    index.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {index.relative_to(OUT_DIR.parents[2])}")
    print(f"\n{len(fixtures)} fixtures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
