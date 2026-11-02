"""Seeded synthetic HVAC fleet generator.

Why synthetic at all: no public dataset links HVAC telemetry, service history,
BOM, hub stock and supplier lead times together, and A01 needs all five in one
world. The brief allows synthetic data; what makes it credible is (a) the
degradation is driven by physically-motivated stress rather than a random draw,
(b) fault signatures match the textbook direction and rough magnitude for each
fault, and (c) the public Azure PdM set validates the same survival pipeline on
real component-level failure history (see sim/azure_pdm.py).

Design that matters for the models:

  Components accumulate *damage* daily at a rate scaled by thermal, cycling and
  load stress, and fail when damage crosses a Weibull-distributed threshold.
  Two consequences: the same component in Ahmedabad (hot, dry) genuinely fails
  sooner than in Bengaluru, so there is real signal for the survival model to
  learn rather than noise; and the damage fraction doubles as both the health
  index the UI needs and the amplitude of the telemetry signature, so the
  telemetry actually carries the information the model is asked to extract.

Volume control: daily aggregates for the whole fleet (~365k rows at 500x730),
hourly only for a 50-asset demo subset over the last 90 days (~108k rows) to
feed the WebSocket replay. Features are per-component-per-day anyway, so daily
aggregates lose nothing the models use.

Run:
    python -m sim.simulate_fleet                       # defaults: 500 assets, 730 d, seed 42
    python -m sim.simulate_fleet --seed 1337 --out data/synthetic_holdout
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_fn

from sim.catalog import (
    ASSET_TYPES,
    CLIMATE_ZONES,
    COMPONENT_CODE,
    COMPONENTS,
    DESIGN_AMBIENT_C,
    HUB_CLIMATE,
    HUBS,
    NOTE_TEMPLATES,
    PM_NOTE_TEMPLATES,
    REPAIR_KITS,
    SITE_CATEGORIES,
    SITE_NAME_PREFIXES,
    SITE_NAME_SUFFIXES,
    SIZE_CLASSES,
    SIZED_COMPONENTS,
    AssetTypeSpec,
    ComponentSpec,
    build_parts_catalog,
)

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

AS_OF = date(2026, 11, 3)  # "today" for the whole prototype
N_DAYS = 730  # 2 years of history
N_ASSETS = 500
N_SITES = 60
HOURLY_ASSETS = 50  # demo subset for the live-stream visual
HOURLY_DAYS = 90
PM_INTERVAL_DAYS = 91  # quarterly preventive visits

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic"


# --------------------------------------------------------------------------
# Operating conditions
# --------------------------------------------------------------------------


def ambient_series(zone: str, start: date, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Daily mean ambient temperature: seasonal sinusoid + weather noise."""
    mean_c, season_amp, _daily_swing, peak_doy = CLIMATE_ZONES[zone]
    doy = np.array([(start + timedelta(days=int(i))).timetuple().tm_yday for i in range(n_days)])
    seasonal = mean_c + season_amp * np.cos(2 * np.pi * (doy - peak_doy) / 365.25)
    # AR(1) weather noise so hot spells persist for days instead of flickering
    noise = np.zeros(n_days)
    eps = rng.normal(0.0, 1.9, n_days)
    for i in range(1, n_days):
        noise[i] = 0.72 * noise[i - 1] + eps[i]
    return seasonal + noise


def ambient_max_series(daily_mean: np.ndarray, zone: str, rng: np.random.Generator) -> np.ndarray:
    """Daily max, used for degree-hours above the design point."""
    _, _, daily_swing, _ = CLIMATE_ZONES[zone]
    return daily_mean + daily_swing / 2.0 + rng.normal(0.0, 0.8, len(daily_mean))


def load_series(
    nominal_duty_h: float,
    ambient: np.ndarray,
    start: date,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duty hours, load fraction and starts/day from schedule + weather.

    Load tracks cooling demand (ambient above ~22 C), clipped to the site's
    nominal duty. Weekends drop for office-like schedules. Starts/day rises as
    load *falls* - a lightly loaded unit short-cycles, which is exactly the
    stress that kills contactors and capacitors.
    """
    n = len(ambient)
    dows = np.array([(start + timedelta(days=int(i))).weekday() for i in range(n)])
    weekend = np.isin(dows, (5, 6))

    cooling_demand = np.clip((ambient - 22.0) / 12.0, 0.05, 1.25)
    load_frac = np.clip(cooling_demand * rng.normal(1.0, 0.07, n), 0.05, 1.0)

    duty = nominal_duty_h * np.clip(cooling_demand, 0.25, 1.0)
    if nominal_duty_h < 20:  # 24/7 sites (hospital, data centre) don't drop at weekends
        duty = np.where(weekend, duty * 0.55, duty)
    duty = np.clip(duty * rng.normal(1.0, 0.05, n), 0.0, 24.0)

    # Short-cycling: many short runs when lightly loaded, few long runs when heavy
    starts = np.clip(duty / np.clip(load_frac, 0.1, 1.0) * rng.normal(1.3, 0.25, n), 0, 90)
    return duty, load_frac, np.round(starts)


def stress_multiplier(
    spec: ComponentSpec,
    ambient_max: np.ndarray,
    load_frac: np.ndarray,
    starts: np.ndarray,
) -> np.ndarray:
    """How much faster than nominal this component ages, day by day.

    Each stress term is normalised so 0 means "nominal duty" and 1 means
    "hard service"; the component's sensitivity then scales its contribution.
    Thermal stress uses degree-hours above the design ambient, which is the
    standard way HVAC derating is expressed.
    """
    therm = np.clip((ambient_max - DESIGN_AMBIENT_C) / 10.0, 0.0, 2.0)
    cyc = np.clip((starts - 12.0) / 24.0, 0.0, 2.0)
    load = np.clip((load_frac - 0.55) / 0.45, 0.0, 1.0)
    return 1.0 + spec.thermal_sens * therm + spec.cycling_sens * cyc + spec.load_sens * load


# --------------------------------------------------------------------------
# Component degradation
# --------------------------------------------------------------------------


def draw_threshold(spec: ComponentSpec, rng: np.random.Generator) -> float:
    """Weibull damage-at-failure threshold, normalised to mean 1.0.

    Scaling by 1/Gamma(1 + 1/k) makes E[threshold] = 1, so under nominal
    stress the expected life equals spec.life_days. The shape parameter then
    controls the *spread*: k ~ 3 (contactors) is a tight wear-out band, k ~ 1.4
    (refrigerant leaks) is close to random.
    """
    k = spec.weibull_shape
    scale = 1.0 / gamma_fn(1.0 + 1.0 / k)
    return float(scale * rng.weibull(k))


@dataclass
class ComponentLife:
    damage_frac: np.ndarray  # health proxy in [0, 1]; 1 = failed
    failure_days: list[int]  # day indices where it failed
    install_day: list[int]  # day index each life started (negative = before window)
    final_threshold: float
    final_cum: float
    final_rate: float  # mean daily damage over the trailing 30 d, for truth TTM


def simulate_component(
    spec: ComponentSpec,
    stress_mult: np.ndarray,
    duty: np.ndarray,
    nominal_duty_h: float,
    init_damage_frac: float,
    rng: np.random.Generator,
    pm_days: np.ndarray,
    target_ttm_days: float | None = None,
) -> ComponentLife:
    """Accumulate damage, fail on threshold crossing, replace, repeat.

    Vectorised per life rather than per day: a cumulative sum plus a
    searchsorted finds the crossing in one shot, and most components see only
    0-3 lives across a 2-year window, so the Python loop is tiny.

    `target_ttm_days` back-solves the starting damage so the component lands on
    a chosen remaining life at the *end* of the window. Needed because setting
    initial damage directly does not control the end state - a compressor with
    a 4000-day life accrues ~0.2 of its threshold over a 730-day window, so a
    guessed starting value lands hundreds of days away from the intended one.
    Used only to place the planted demo scenario; the rest of the fleet draws
    its starting damage at random.
    """
    n = len(stress_mult)
    base_rate = 1.0 / spec.life_days
    # Components that age on runtime scale with duty; calendar-aged parts
    # (control boards, capacitors) barely care how much the machine ran.
    runtime_factor = spec.runtime_sens * (duty / max(nominal_duty_h, 1e-6)) + (1.0 - spec.runtime_sens)
    inc = base_rate * stress_mult * np.clip(runtime_factor, 0.0, 3.0)

    damage = np.zeros(n)
    failure_days: list[int] = []
    install_day: list[int] = [-1]  # the life in progress at window start

    thresh = draw_threshold(spec, rng)

    if target_ttm_days is not None:
        # TTM at window end = (thresh - cum_end) / rate, so the damage we want
        # to finish on is  1 - target_ttm * rate / thresh. Subtract everything
        # the component will accrue during the window to get where it must start.
        trailing_rate = float(np.mean(inc[-30:])) if n >= 30 else float(np.mean(inc))
        target_final_frac = 1.0 - (target_ttm_days * max(trailing_rate, 1e-12)) / thresh
        accrual_frac = float(inc.sum()) / thresh
        solved = target_final_frac - accrual_frac
        # A negative solution means this part cannot span the window on one
        # life (short-lived consumables); let it renew normally instead.
        init_damage_frac = solved if solved > 0.0 else max(0.0, target_final_frac * 0.5)

    cum = float(np.clip(init_damage_frac, 0.0, 0.995)) * thresh
    day = 0

    while day < n:
        cs = np.cumsum(inc[day:])
        remaining = thresh - cum
        idx = int(np.searchsorted(cs, remaining))
        if idx >= len(cs):
            damage[day:] = np.minimum((cum + cs) / thresh, 1.0)
            cum = cum + float(cs[-1])
            break
        fail_day = day + idx
        damage[day : fail_day + 1] = np.minimum((cum + cs[: idx + 1]) / thresh, 1.0)
        failure_days.append(fail_day)
        # Replaced same or next day; a fresh part means a fresh threshold.
        thresh = draw_threshold(spec, rng)
        cum = 0.0
        day = fail_day + 1
        if day < n:
            install_day.append(day)

    # PM visits replace consumables (filters, belts) regardless of damage.
    if spec.pm_replaceable and len(pm_days):
        for pm_day in pm_days:
            pm_day = int(pm_day)
            if pm_day <= 0 or pm_day >= n:
                continue
            if damage[pm_day] > 0.25:  # only if meaningfully worn
                drop = damage[pm_day]
                damage[pm_day:] = np.maximum(damage[pm_day:] - drop, 0.0)
                cum = max(cum - drop * thresh, 0.0)
                install_day.append(pm_day)

    trailing = inc[max(n - 30, 0) :]
    return ComponentLife(
        damage_frac=damage,
        failure_days=failure_days,
        install_day=install_day,
        final_threshold=thresh,
        final_cum=cum,
        final_rate=float(np.mean(trailing)) if len(trailing) else base_rate,
    )


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------

# Nominal healthy signal values. Pressures in kPa, temps in C, current in A.
# Refrigerant-side signals are None for air-only machines (AHU).
BASE_SIGNALS: dict[str, dict[str, float | None]] = {
    "chiller": {
        "suction_p": 880.0, "discharge_p": 2400.0, "superheat": 5.5, "subcool": 4.5,
        "supply_t": 7.0, "return_t": 12.5, "comp_current": 40.0, "vib_rms": 1.8,
        "vib_kurtosis": 3.0, "current_imbalance": 1.2, "cop": 3.20, "charge_frac": 1.0,
    },
    "rtu": {
        "suction_p": 790.0, "discharge_p": 2250.0, "superheat": 6.0, "subcool": 5.0,
        "supply_t": 13.0, "return_t": 24.0, "comp_current": 18.0, "vib_rms": 2.2,
        "vib_kurtosis": 3.1, "current_imbalance": 1.5, "cop": 3.00, "charge_frac": 1.0,
    },
    "ahu": {
        "suction_p": None, "discharge_p": None, "superheat": None, "subcool": None,
        "supply_t": 14.0, "return_t": 24.5, "comp_current": None, "vib_rms": 2.4,
        "vib_kurtosis": 3.2, "current_imbalance": 1.3, "cop": None, "charge_frac": None,
    },
    "split_heat_pump": {
        "suction_p": 820.0, "discharge_p": 2300.0, "superheat": 6.5, "subcool": 4.0,
        "supply_t": 14.0, "return_t": 25.0, "comp_current": 9.5, "vib_rms": 2.0,
        "vib_kurtosis": 3.0, "current_imbalance": 1.6, "cop": 3.35, "charge_frac": 1.0,
    },
    "vrf": {
        "suction_p": 810.0, "discharge_p": 2320.0, "superheat": 6.2, "subcool": 4.2,
        "supply_t": 13.5, "return_t": 24.0, "comp_current": 22.0, "vib_rms": 1.9,
        "vib_kurtosis": 3.0, "current_imbalance": 1.4, "cop": 3.25, "charge_frac": 1.0,
    },
}

SIGNAL_NOISE = {
    "suction_p": 9.0, "discharge_p": 22.0, "superheat": 0.45, "subcool": 0.35,
    "supply_t": 0.30, "return_t": 0.40, "comp_current": 0.55, "vib_rms": 0.14,
    "vib_kurtosis": 0.22, "current_imbalance": 0.18, "cop": 0.035, "charge_frac": 0.012,
}

SIGNALS = list(SIGNAL_NOISE.keys())


def build_telemetry(
    atype: AssetTypeSpec,
    capacity_kw: float,
    damage: dict[str, np.ndarray],
    ambient: np.ndarray,
    ambient_max: np.ndarray,
    duty: np.ndarray,
    load_frac: np.ndarray,
    starts: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Healthy baseline + fault signatures + noise + dropouts + sensor bias.

    Signature contributions are linear in each component's damage fraction and
    additive across components, which keeps the inverse problem learnable while
    still producing the right qualitative pattern: fouling raises head pressure
    and drops COP, a leak drops suction and raises superheat, wear raises
    current and vibration.
    """
    n = len(ambient)
    base = BASE_SIGNALS[atype.key]
    out: dict[str, np.ndarray] = {}

    # Ambient pushes the condensing side: hotter day -> higher head pressure.
    amb_rise = np.clip(ambient_max - 30.0, -10.0, 20.0)

    for sig in SIGNALS:
        b = base.get(sig)
        if b is None:
            out[sig] = np.full(n, np.nan)
            continue

        series = np.full(n, float(b))

        if sig == "discharge_p":
            series = series + 26.0 * amb_rise + 95.0 * (load_frac - 0.55)
        elif sig == "suction_p":
            series = series + 22.0 * (load_frac - 0.55)
        elif sig == "comp_current":
            series = series * (0.78 + 0.34 * load_frac) * (1.0 + 0.010 * amb_rise)
        elif sig == "cop":
            # COP falls as condensing temperature rises - basic Carnot pressure
            series = series * (1.0 - 0.0135 * np.clip(amb_rise, 0, None))
        elif sig == "supply_t":
            series = series + 0.5 * (load_frac - 0.55)
        elif sig == "return_t":
            series = series + 0.08 * amb_rise

        # Fault signatures, scaled by how far each component has degraded.
        for ckey, dmg in damage.items():
            delta = COMPONENTS[ckey].signature.get(sig)
            if delta:
                series = series + delta * dmg

        series = series + rng.normal(0.0, SIGNAL_NOISE[sig], n)

        # A persistent sensor bias on ~8% of channels: the model has to cope
        # with miscalibration, and the data-quality panel has something to show.
        if rng.random() < 0.08:
            series = series + rng.normal(0.0, SIGNAL_NOISE[sig] * 3.5)

        out[sig] = series

    # Dropouts: short gateway outages. Runs of 1-9 days, ~4% of days overall.
    gap_mask = np.zeros(n, dtype=bool)
    n_gaps = rng.poisson(3.2)
    for _ in range(int(n_gaps)):
        g_start = int(rng.integers(0, n))
        g_len = int(rng.integers(1, 10))
        gap_mask[g_start : g_start + g_len] = True
    for sig in SIGNALS:
        out[sig][gap_mask] = np.nan

    out["ambient_t"] = ambient
    out["ambient_t_max"] = ambient_max
    out["runtime_h"] = duty
    out["load_frac"] = load_frac
    out["starts"] = starts
    out["short_cycles"] = np.clip(starts - 18.0, 0, None)
    return out


# --------------------------------------------------------------------------
# World construction
# --------------------------------------------------------------------------


def build_sites(rng: np.random.Generator, n_sites: int) -> pd.DataFrame:
    cats, crits, shares, duties = zip(*SITE_CATEGORIES)
    probs = np.array(shares) / np.sum(shares)
    rows = []
    used: set[str] = set()
    for i in range(n_sites):
        ci = int(rng.choice(len(cats), p=probs))
        hub_id, hub_name, region, lat, lon = HUBS[int(rng.integers(0, len(HUBS)))]
        for _ in range(40):
            name = f"{SITE_NAME_PREFIXES[int(rng.integers(0, len(SITE_NAME_PREFIXES)))]} " \
                   f"{SITE_NAME_SUFFIXES[int(rng.integers(0, len(SITE_NAME_SUFFIXES)))]}"
            if name not in used:
                used.add(name)
                break
        rows.append(
            {
                "site_id": f"SITE-{i + 1:03d}",
                "name": name,
                "category": cats[ci],
                "criticality": crits[ci],
                "nominal_duty_h": duties[ci],
                "hub_id": hub_id,
                "region": region,
                # Scatter sites around their hub so the map isn't 8 dots
                "lat": lat + float(rng.normal(0, 0.28)),
                "lon": lon + float(rng.normal(0, 0.28)),
                "climate_zone": HUB_CLIMATE[hub_id],
            }
        )
    return pd.DataFrame(rows)


def size_class_of(atype: AssetTypeSpec, capacity_kw: float) -> str:
    lo, hi = atype.size_class_edges
    return "S" if capacity_kw < lo else ("M" if capacity_kw < hi else "L")


def build_assets(rng: np.random.Generator, sites: pd.DataFrame, n_assets: int) -> pd.DataFrame:
    shares = np.array([t.share for t in ASSET_TYPES])
    probs = shares / shares.sum()
    rows = []
    for i in range(n_assets):
        t = ASSET_TYPES[int(rng.choice(len(ASSET_TYPES), p=probs))]
        site = sites.iloc[int(rng.integers(0, len(sites)))]
        cap = float(rng.uniform(*t.capacity_range_kw))
        # Fleet age: mostly 1-12 years, skewed young
        age_years = float(np.clip(rng.gamma(2.4, 2.2), 0.4, 14.0))
        install = AS_OF - timedelta(days=int(age_years * 365.25))
        prefix = {"chiller": "CH", "rtu": "RTU", "ahu": "AHU", "split_heat_pump": "SHP", "vrf": "VRF"}[t.key]
        rows.append(
            {
                "asset_id": f"{prefix}-{i + 1:03d}",
                "asset_label": f"{t.label} {prefix}-{i + 1:03d}",
                "asset_type": t.key,
                "model": t.models[int(rng.integers(0, len(t.models)))],
                "refrigerant": t.refrigerant,
                "site_id": site["site_id"],
                "site_name": site["name"],
                "site_category": site["category"],
                "hub_id": site["hub_id"],
                "region": site["region"],
                "lat": site["lat"],
                "lon": site["lon"],
                "climate_zone": site["climate_zone"],
                "capacity_kw": round(cap, 1),
                "size_class": size_class_of(t, cap),
                "install_date": install,
                "age_days": int(age_years * 365.25),
                "nominal_duty_h": site["nominal_duty_h"],
                "criticality": site["criticality"],
            }
        )
    return pd.DataFrame(rows)


def part_for(component_type: str, size_class: str, refrigerant: str) -> str | None:
    """Resolve a component to its part number at this asset's size."""
    from sim.catalog import _PART_PREFIX

    if component_type == "refrigerant_circuit":
        return None
    prefix = _PART_PREFIX[component_type]
    if component_type in SIZED_COMPONENTS:
        return f"{prefix}-{size_class}"
    return prefix


def resolve_kit_part(token: str, component_type: str, size_class: str, refrigerant: str) -> str | None:
    """Expand a repair-kit token into a concrete part number."""
    if token == "@self":
        return part_for(component_type, size_class, refrigerant)
    if token == "@refrigerant":
        return "REF-R32-KG" if refrigerant == "R32" else ("REF-R410A-KG" if refrigerant == "R410A" else None)
    if token in {"BELT", "FLT"}:  # sized consumables
        return f"{token}-{size_class}"
    return token


# --------------------------------------------------------------------------
# Main simulation
# --------------------------------------------------------------------------


def simulate(
    seed: int = 42,
    n_assets: int = N_ASSETS,
    n_days: int = N_DAYS,
    n_sites: int = N_SITES,
    plant_hero: bool = True,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = AS_OF - timedelta(days=n_days - 1)
    dates = pd.to_datetime([start + timedelta(days=i) for i in range(n_days)])

    sites = build_sites(rng, n_sites)
    assets = build_assets(rng, sites, n_assets)

    # ---- the planted demo scenario -------------------------------------
    # Deliberately chosen *initial conditions*, not a faked prediction: one
    # Bengaluru chiller starts the window with advanced condenser fouling and a
    # worn compressor, plus two more nearby. The physics and the trained model
    # then produce the TTM numbers on their own. Documented here and in the
    # pitch so nobody has to guess what is scenario and what is model.
    hero_ids: list[str] = []
    if plant_hero:
        blr_chillers = assets[
            (assets["hub_id"] == "HUB-BLR") & (assets["asset_type"] == "chiller")
        ].index.tolist()
        if len(blr_chillers) < 3:  # top up by relocating chillers to Bengaluru
            spare = assets[(assets["asset_type"] == "chiller")].index.tolist()
            for idx in spare:
                if len(blr_chillers) >= 3:
                    break
                if idx not in blr_chillers:
                    assets.loc[idx, ["hub_id", "region", "climate_zone"]] = ["HUB-BLR", "South", "2A"]
                    blr_chillers.append(idx)
        for idx in blr_chillers[:3]:
            assets.loc[idx, "criticality"] = "A"
            # Same size class so all three draw the same compressor part number -
            # otherwise the three urgent jobs need three different parts and the
            # "one transfer covers them" story falls apart.
            assets.loc[idx, "size_class"] = "L"
            assets.loc[idx, "capacity_kw"] = float(
                np.round(assets.loc[idx, "capacity_kw"] if assets.loc[idx, "capacity_kw"] > 450 else 470.0, 1)
            )
        hero_ids = [assets.loc[i, "asset_id"] for i in blr_chillers[:3]]

    # The planted scenario, expressed as target remaining life at AS_OF rather
    # than as starting damage (see simulate_component). Three Bengaluru chillers
    # need the same compressor inside the urgent window, with the lead asset's
    # condenser fouling well advanced so the Asset Twin shows the *cause* and
    # not just the symptom.
    hero_targets: dict[int, dict[str, float]] = {
        0: {"compressor": 9.0, "condenser_coil": 55.0},
        1: {"compressor": 16.0},
        2: {"compressor": 24.0},
    }

    comp_rows: list[dict] = []
    truth_rows: list[dict] = []
    wo_rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    hourly_frames: list[pd.DataFrame] = []

    atype_by_key = {t.key: t for t in ASSET_TYPES}
    hourly_pick = set(
        assets["asset_id"].iloc[
            rng.choice(len(assets), size=min(HOURLY_ASSETS, len(assets)), replace=False)
        ]
    )
    # Keep the hero assets in the hourly subset so the demo stream shows them.
    hourly_pick.update(hero_ids)

    wo_counter = 0

    for row_i, asset in assets.iterrows():
        atype = atype_by_key[asset["asset_type"]]
        zone = asset["climate_zone"]
        a_rng = np.random.default_rng(seed * 100_003 + int(row_i))

        ambient = ambient_series(zone, start, n_days, a_rng)
        amb_max = ambient_max_series(ambient, zone, a_rng)
        duty, load_frac, starts = load_series(float(asset["nominal_duty_h"]), ambient, start, a_rng)

        # Quarterly PM visits, offset per asset so the service load is spread
        pm_offset = int(a_rng.integers(0, PM_INTERVAL_DAYS))
        pm_days = np.arange(pm_offset, n_days, PM_INTERVAL_DAYS)

        damage: dict[str, np.ndarray] = {}
        lives: dict[str, ComponentLife] = {}

        for ckey in atype.component_types:
            spec = COMPONENTS[ckey]
            smult = stress_multiplier(spec, amb_max, load_frac, starts)

            # Where this part sits in its life at window start. Drawn from an
            # equilibrium-renewal-ish uniform so the fleet starts with a
            # realistic spread of health rather than all-new.
            hero_slot = hero_ids.index(asset["asset_id"]) if asset["asset_id"] in hero_ids else None
            target_ttm = hero_targets.get(hero_slot, {}).get(ckey) if hero_slot is not None else None
            max_frac = min(1.0, float(asset["age_days"]) / spec.life_days)
            init = float(a_rng.uniform(0.0, max(max_frac, 0.05)) ** 0.85)

            life = simulate_component(
                spec, smult, duty, float(asset["nominal_duty_h"]), init, a_rng, pm_days,
                target_ttm_days=target_ttm,
            )
            damage[ckey] = life.damage_frac
            lives[ckey] = life

            component_id = f"{asset['asset_id']}-{COMPONENT_CODE[ckey]}"
            last_install_idx = max([d for d in life.install_day if d < n_days], default=-1)
            last_service = dates[last_install_idx] if last_install_idx >= 0 else pd.Timestamp(asset["install_date"])

            comp_rows.append(
                {
                    "component_id": component_id,
                    "asset_id": asset["asset_id"],
                    "component_type": ckey,
                    "criticality": spec.criticality,
                    "install_date": last_service,
                    "last_service_date": last_service,
                    "part_no": part_for(ckey, asset["size_class"], asset["refrigerant"]),
                    "n_failures_in_window": len(life.failure_days),
                }
            )

            # Ground truth remaining life at AS_OF. Written to its own table and
            # NEVER joined into the feature matrix - it exists so the backtest
            # can be scored honestly, not so the model can cheat.
            remaining_damage = max(life.final_threshold - life.final_cum, 0.0)
            rate = max(life.final_rate, 1e-9)
            truth_rows.append(
                {
                    "component_id": component_id,
                    "asset_id": asset["asset_id"],
                    "component_type": ckey,
                    "true_ttm_days": float(min(remaining_damage / rate, 5000.0)),
                    "damage_frac_now": float(life.damage_frac[-1]),
                }
            )

            # Corrective work orders at each failure
            for fd in life.failure_days:
                modes, probs = zip(*spec.failure_modes)
                mode = str(a_rng.choice(modes, p=np.array(probs) / np.sum(probs)))
                kit = REPAIR_KITS.get((ckey, mode), [])
                used: list[str] = []
                qty: list[float] = []
                for token, q, p_used in kit:
                    if a_rng.random() <= p_used:
                        pn = resolve_kit_part(token, ckey, asset["size_class"], asset["refrigerant"])
                        if pn:
                            used.append(pn)
                            qty.append(float(q))
                # First-time fix fails when a part wasn't on the van
                ftf = bool(a_rng.random() < (0.62 if spec.criticality == "A" else 0.78))
                wo_counter += 1
                is_emergency = spec.criticality == "A" and a_rng.random() < 0.55
                open_ts = dates[fd]
                wo_rows.append(
                    {
                        "wo_id": f"WO-{wo_counter:05d}",
                        "asset_id": asset["asset_id"],
                        "component_id": component_id,
                        "component_type": ckey,
                        "type": "emergency" if is_emergency else "CM",
                        "open_ts": open_ts,
                        "close_ts": open_ts + pd.Timedelta(days=int(a_rng.integers(0, 4))),
                        "failure_mode": mode,
                        "parts_used": used,
                        "parts_qty": qty,
                        "first_time_fix": ftf,
                        "hub_id": asset["hub_id"],
                        "notes": str(a_rng.choice(NOTE_TEMPLATES.get(mode, ("Repair completed.",)))),
                        "downtime_h": float(np.round(a_rng.gamma(2.0, 3.5 if is_emergency else 1.6), 1)),
                    }
                )

        # Preventive work orders
        for pm_day in pm_days:
            if pm_day <= 0 or pm_day >= n_days:
                continue
            wo_counter += 1
            pm_parts = [
                p for p in (
                    part_for("filter", asset["size_class"], asset["refrigerant"])
                    if "filter" in atype.component_types else None,
                    f"BELT-{asset['size_class']}" if "belt" in atype.component_types else None,
                ) if p
            ]
            open_ts = dates[int(pm_day)]
            wo_rows.append(
                {
                    "wo_id": f"WO-{wo_counter:05d}",
                    "asset_id": asset["asset_id"],
                    "component_id": None,
                    "component_type": None,
                    "type": "PM",
                    "open_ts": open_ts,
                    "close_ts": open_ts,
                    "failure_mode": None,
                    "parts_used": pm_parts,
                    "parts_qty": [1.0] * len(pm_parts),
                    "first_time_fix": True,
                    "hub_id": asset["hub_id"],
                    "notes": str(a_rng.choice(PM_NOTE_TEMPLATES)),
                    "downtime_h": 0.0,
                }
            )

        tele = build_telemetry(
            atype, float(asset["capacity_kw"]), damage, ambient, amb_max, duty, load_frac, starts, a_rng
        )
        df = pd.DataFrame(tele)
        df.insert(0, "asset_id", asset["asset_id"])
        df.insert(0, "ts", dates)
        daily_frames.append(df)

        # Per-component health, kept daily so the twin schematic and the
        # residual feature have something to read.
        for ckey, dmg in damage.items():
            df[f"health_{ckey}"] = 1.0 - dmg

        if asset["asset_id"] in hourly_pick:
            hourly_frames.append(
                _expand_hourly(df.tail(HOURLY_DAYS), asset["asset_id"], a_rng)
            )

    telemetry_daily = pd.concat(daily_frames, ignore_index=True)
    telemetry_hourly = (
        pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame()
    )
    components = pd.DataFrame(comp_rows)
    truth = pd.DataFrame(truth_rows)
    work_orders = pd.DataFrame(wo_rows)

    parts, suppliers, bom = build_supply_tables(rng, assets, components)
    inventory = build_inventory(rng, assets, components, work_orders, parts, hero_ids)
    hubs_df = build_hubs_table()

    return {
        "hubs": hubs_df,
        "sites": sites,
        "assets": assets,
        "components": components,
        "components_truth": truth,
        "telemetry_daily": telemetry_daily,
        "telemetry_hourly": telemetry_hourly,
        "work_orders": work_orders,
        "parts": parts,
        "suppliers": suppliers,
        "bom": bom,
        "inventory": inventory,
    }


def _expand_hourly(daily: pd.DataFrame, asset_id: str, rng: np.random.Generator) -> pd.DataFrame:
    """Interpolate a daily tail to hourly for the live-replay visual.

    Only the signals the Command Center animates; adds a diurnal shape so the
    stream looks like a real machine cycling rather than a straight line.
    """
    keep = ["suction_p", "discharge_p", "superheat", "comp_current", "vib_rms", "cop", "ambient_t"]
    keep = [k for k in keep if k in daily.columns]
    rows = []
    hours = np.arange(24)
    diurnal = np.sin((hours - 6) / 24.0 * 2 * np.pi)
    for _, d in daily.iterrows():
        base_ts = pd.Timestamp(d["ts"])
        frame = {"ts": [base_ts + pd.Timedelta(hours=int(h)) for h in hours], "asset_id": asset_id}
        for k in keep:
            v = d[k]
            if pd.isna(v):
                frame[k] = [np.nan] * 24
                continue
            amp = abs(v) * (0.035 if k != "ambient_t" else 0.0) + (3.5 if k == "ambient_t" else 0.0)
            frame[k] = v + amp * diurnal + rng.normal(0, abs(v) * 0.006 + 1e-6, 24)
        rows.append(pd.DataFrame(frame))
    return pd.concat(rows, ignore_index=True)


def build_hubs_table() -> pd.DataFrame:
    rows = []
    for hub_id, name, region, lat, lon in HUBS:
        rows.append(
            {
                "hub_id": hub_id, "name": name, "region": region,
                "lat": lat, "lon": lon, "climate_zone": HUB_CLIMATE[hub_id],
            }
        )
    # Pairwise transfer times live in their own table (transfer_matrix) rather
    # than in df.attrs - Parquet metadata is JSON, so a tuple-keyed dict cannot
    # round-trip through it.
    return pd.DataFrame(rows)


def transfer_matrix() -> pd.DataFrame:
    rows = []
    for a in HUBS:
        for b in HUBS:
            if a[0] == b[0]:
                continue
            d_km = _haversine(a[3], a[4], b[3], b[4])
            rows.append(
                {
                    "from_hub_id": a[0],
                    "to_hub_id": b[0],
                    "distance_km": round(d_km, 1),
                    "transfer_days": round(min(1.0 + d_km / 700.0, 4.5), 1),
                    "same_region": a[2] == b[2],
                }
            )
    return pd.DataFrame(rows)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_supply_tables(
    rng: np.random.Generator, assets: pd.DataFrame, components: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parts master, supplier lead times, and the BOM/repair-kit expansion."""
    catalog = build_parts_catalog()
    parts_rows, sup_rows = [], []
    for i, p in enumerate(catalog):
        supplier_id = f"SUP-{(i % 9) + 1:03d}"
        parts_rows.append(
            {
                "part_no": p.part_no,
                "description": p.description,
                "unit_cost": p.unit_cost,
                "criticality": p.criticality,
                "shelf_life_days": p.shelf_life_days,
                "supplier_id": supplier_id,
            }
        )
        # Per-part lead time jitter so two parts from one supplier can differ
        sup_rows.append(
            {
                "supplier_id": supplier_id,
                "part_no": p.part_no,
                "lead_time_mean_d": round(float(p.lead_time_mean_d * rng.normal(1.0, 0.09)), 1),
                "lead_time_sd_d": round(float(p.lead_time_sd_d * rng.normal(1.0, 0.14)), 2),
                "moq": p.moq,
                "expedite_cost": round(p.unit_cost * (p.expedite_cost_mult - 1.0), 2),
                # Occasional chronic late supplier, so the scenario screen has
                # a supplier-delay story that is already in the data
                "on_time_rate": round(float(np.clip(rng.beta(8, 2), 0.45, 0.99)), 3),
            }
        )

    # BOM: asset_model x component_type x failure_mode -> part, qty, p_used
    bom_rows = []
    model_sizes = assets[["model", "size_class", "refrigerant"]].drop_duplicates()
    for _, ms in model_sizes.iterrows():
        for (ckey, mode), kit in REPAIR_KITS.items():
            for token, q, p_used in kit:
                pn = resolve_kit_part(token, ckey, ms["size_class"], ms["refrigerant"])
                if not pn:
                    continue
                bom_rows.append(
                    {
                        "asset_model": ms["model"],
                        "size_class": ms["size_class"],
                        "component_type": ckey,
                        "failure_mode": mode,
                        "part_no": pn,
                        "qty_per_repair": q,
                        "p_used_truth": p_used,  # generator ground truth; planning re-derives
                        "substitute_part_no": None,
                    }
                )
    return pd.DataFrame(parts_rows), pd.DataFrame(sup_rows), pd.DataFrame(bom_rows)


def build_inventory(
    rng: np.random.Generator,
    assets: pd.DataFrame,
    components: pd.DataFrame,
    work_orders: pd.DataFrame,
    parts: pd.DataFrame,
    hero_ids: list[str],
) -> pd.DataFrame:
    """On-hand stock per hub, set from historical consumption with noise.

    Deliberately *mis*-positioned: stock is sized on each hub's own past
    consumption with a wide multiplier, which is exactly what a min/max policy
    on history produces - some hubs sitting on excess, others short of what the
    condition-based forecast says they will need. That gap is the problem the
    prototype exists to close, so it has to be present in the starting state.
    """
    # Historical consumption per hub x part
    cons = []
    for _, wo in work_orders.iterrows():
        for pn, q in zip(wo["parts_used"] or [], wo["parts_qty"] or []):
            cons.append({"hub_id": wo["hub_id"], "part_no": pn, "qty": q})
    cons_df = (
        pd.DataFrame(cons).groupby(["hub_id", "part_no"], as_index=False)["qty"].sum()
        if cons
        else pd.DataFrame(columns=["hub_id", "part_no", "qty"])
    )

    hub_ids = [h[0] for h in HUBS]
    rows = []
    cost_by_part = dict(zip(parts["part_no"], parts["unit_cost"]))
    for hub_id in hub_ids:
        sub = cons_df[cons_df["hub_id"] == hub_id]
        hist = dict(zip(sub["part_no"], sub["qty"]))
        for pn in parts["part_no"]:
            annual = hist.get(pn, 0.0) / 2.0  # window is 2 years
            # min/max from history with a deliberately crude multiplier
            mult = float(rng.uniform(0.35, 2.4))
            target = annual * mult / 12.0 * 3.0  # ~3 months of cover
            on_hand = int(max(0, round(target + rng.normal(0, max(target * 0.4, 0.6)))))
            # Expensive parts are stocked thin regardless of need
            if cost_by_part.get(pn, 0) > 40_000:
                on_hand = min(on_hand, int(rng.integers(0, 3)))
            rows.append(
                {
                    "hub_id": hub_id,
                    "part_no": pn,
                    "on_hand": on_hand,
                    "on_order": int(rng.integers(0, 2)) if rng.random() < 0.12 else 0,
                    "reserved": 0,
                    "min_qty": int(max(0, round(target * 0.5))),
                    "max_qty": int(max(1, round(target * 1.6))),
                }
            )
    inv = pd.DataFrame(rows)

    # Planted scenario: Bengaluru short of the compressor its urgent jobs need,
    # Chennai holding well above its own requirement. Makes "transfer, don't
    # expedite" the genuinely cheapest correct answer rather than a contrivance.
    if hero_ids:
        hero_assets = assets[assets["asset_id"].isin(hero_ids)]
        if len(hero_assets):
            size = hero_assets.iloc[0]["size_class"]
            cmp_part = f"CMP-{size}"
            inv.loc[(inv["hub_id"] == "HUB-BLR") & (inv["part_no"] == cmp_part), "on_hand"] = 1
            inv.loc[(inv["hub_id"] == "HUB-MAA") & (inv["part_no"] == cmp_part), "on_hand"] = 5
    return inv


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def write_outputs(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = dict(tables)
    tables["hub_transfers"] = transfer_matrix()
    for name, df in tables.items():
        if df is None or len(df) == 0:
            continue
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {name:20s} {len(df):>9,} rows  -> {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the ServiceSense synthetic fleet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--assets", type=int, default=N_ASSETS)
    ap.add_argument("--days", type=int, default=N_DAYS)
    ap.add_argument("--sites", type=int, default=N_SITES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--no-hero",
        action="store_true",
        help="Skip the planted demo scenario (use for honest holdout generation)",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    print(f"Simulating {args.assets} assets x {args.days} days, seed={args.seed} ...")
    tables = simulate(
        seed=args.seed,
        n_assets=args.assets,
        n_days=args.days,
        n_sites=args.sites,
        plant_hero=not args.no_hero,
    )
    print(f"Simulated in {time.perf_counter() - t0:.1f}s. Writing to {args.out} ...")
    write_outputs(tables, args.out)

    wo = tables["work_orders"]
    cm = wo[wo["type"] != "PM"]
    truth = tables["components_truth"]
    print(
        f"\n  components:        {len(tables['components']):,}"
        f"\n  failures in window:{len(cm):,}"
        f"\n  PM visits:         {len(wo) - len(cm):,}"
        f"\n  urgent now (<=14d):{int((truth['true_ttm_days'] <= 14).sum()):,}"
        f"\n  planned (15-90d):  {int(((truth['true_ttm_days'] > 14) & (truth['true_ttm_days'] <= 90)).sum()):,}"
        f"\n  total time:        {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
