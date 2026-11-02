"""Static catalogue: asset types, components, parts, repair kits, hubs, climate.

Separated from the generator so the *world definition* can be read and argued
about without wading through simulation code. Everything here is a plain
dict/dataclass - no randomness, no I/O.

The repair kits in REPAIR_KITS are the generator's ground truth. The planning
layer deliberately does NOT import them: planning/repair_kits.py re-derives
P(used) from the simulated work-order history, exactly as it would have to on
real data. If the two ever disagree, that is a bug worth knowing about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Service network
# --------------------------------------------------------------------------

# hub_id, name, region, lat, lon
HUBS: list[tuple[str, str, str, float, float]] = [
    ("HUB-BLR", "Bengaluru", "South", 12.9716, 77.5946),
    ("HUB-MAA", "Chennai", "South", 13.0827, 80.2707),
    ("HUB-HYD", "Hyderabad", "South", 17.3850, 78.4867),
    ("HUB-PNQ", "Pune", "West", 18.5204, 73.8567),
    ("HUB-BOM", "Mumbai", "West", 19.0760, 72.8777),
    ("HUB-AMD", "Ahmedabad", "West", 23.0225, 72.5714),
    ("HUB-DEL", "Delhi", "North", 28.7041, 77.1025),
    ("HUB-CCU", "Kolkata", "East", 22.5726, 88.3639),
]

# Climate zone per hub. Drives the ambient profile, which drives stress, which
# is what makes TTM genuinely vary across the fleet rather than being noise.
HUB_CLIMATE = {
    "HUB-BLR": "2A",  # moderate tropical savanna
    "HUB-MAA": "1A",  # hot humid
    "HUB-HYD": "1A",
    "HUB-PNQ": "2A",
    "HUB-BOM": "1A",
    "HUB-AMD": "1B",  # hot dry
    "HUB-DEL": "2B",  # hot dry with cold winter
    "HUB-CCU": "1A",
}

# zone -> (annual mean C, seasonal amplitude C, daily swing C, peak day-of-year)
CLIMATE_ZONES: dict[str, tuple[float, float, float, int]] = {
    "1A": (28.5, 4.5, 7.0, 135),
    "1B": (29.0, 7.5, 11.0, 140),
    "2A": (24.5, 4.0, 9.0, 115),
    "2B": (25.5, 9.0, 12.0, 150),
}

# Design ambient: degree-hours above this drive thermal stress.
DESIGN_AMBIENT_C = 35.0

# Site categories: (label, asset criticality, share of sites, duty hours/day)
SITE_CATEGORIES: list[tuple[str, str, float, float]] = [
    ("Hospital", "A", 0.12, 23.0),
    ("Data centre", "A", 0.10, 24.0),
    ("Office park", "B", 0.34, 13.0),
    ("Hotel", "B", 0.16, 18.0),
    ("Retail mall", "C", 0.20, 14.0),
    ("Warehouse", "C", 0.08, 9.0),
]

SITE_NAME_PREFIXES = [
    "Prestige", "Embassy", "RMZ", "Cessna", "Brigade", "DLF", "Mindspace",
    "Hiranandani", "Phoenix", "Salarpuria", "Godrej", "Oberoi", "Raheja",
    "Nirlon", "Magarpatta", "Bagmane", "Manyata", "Ascendas", "Olympia", "Tidel",
]
SITE_NAME_SUFFIXES = [
    "Tech Park", "Business Bay", "Corporate Centre", "Plaza", "Square",
    "One", "Campus", "Hub", "Towers", "Commercial Centre",
]

# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetTypeSpec:
    key: str
    label: str
    refrigerant: str
    capacity_range_kw: tuple[float, float]
    size_class_edges: tuple[float, float]  # kW cuts for S|M|L part sizing
    models: tuple[str, ...]
    component_types: tuple[str, ...]
    share: float  # share of the installed base


# Component sets differ by machine: a chiller has no belts, an AHU has no
# compressor of its own, a split heat pump has no VFD.
_CHILLER_COMPONENTS = (
    "compressor", "condenser_coil", "condenser_fan_motor", "evaporator",
    "eev", "vfd", "control_board", "run_capacitor", "contactor",
    "refrigerant_circuit", "fan_bearing", "filter",
)
_RTU_COMPONENTS = (
    "compressor", "condenser_coil", "condenser_fan_motor", "evap_blower_motor",
    "eev", "control_board", "run_capacitor", "contactor",
    "refrigerant_circuit", "belt", "fan_bearing", "filter",
)
_AHU_COMPONENTS = (
    "evap_blower_motor", "cooling_coil", "control_board", "contactor",
    "belt", "fan_bearing", "damper_actuator", "filter",
)
_SPLIT_COMPONENTS = (
    "compressor", "condenser_coil", "condenser_fan_motor", "evap_blower_motor",
    "txv", "control_board", "run_capacitor", "contactor",
    "refrigerant_circuit", "fan_bearing", "filter",
)
_VRF_COMPONENTS = (
    "compressor", "condenser_coil", "condenser_fan_motor", "evap_blower_motor",
    "eev", "vfd", "control_board", "contactor",
    "refrigerant_circuit", "fan_bearing", "filter",
)

ASSET_TYPES: list[AssetTypeSpec] = [
    AssetTypeSpec("chiller", "Chiller", "R32", (150.0, 700.0), (250.0, 450.0),
                  ("CGAM-080-R32", "CGAM-130-R32", "RTAC-200-R32"),
                  _CHILLER_COMPONENTS, 0.18),
    AssetTypeSpec("rtu", "Rooftop unit", "R410A", (20.0, 120.0), (40.0, 80.0),
                  ("RTU-07T-R410A", "RTU-12T-R410A", "RTU-20T-R410A"),
                  _RTU_COMPONENTS, 0.32),
    AssetTypeSpec("ahu", "Air handling unit", "none", (10.0, 90.0), (30.0, 60.0),
                  ("AHU-10K-STD", "AHU-25K-STD", "AHU-40K-STD"),
                  _AHU_COMPONENTS, 0.20),
    AssetTypeSpec("split_heat_pump", "Split heat pump", "R32", (5.0, 35.0), (12.0, 22.0),
                  ("SHP-03T-R32", "SHP-05T-R32", "SHP-08T-R32"),
                  _SPLIT_COMPONENTS, 0.18),
    AssetTypeSpec("vrf", "VRF system", "R410A", (30.0, 180.0), (60.0, 120.0),
                  ("VRF-12T-R410A", "VRF-20T-R410A", "VRF-34T-R410A"),
                  _VRF_COMPONENTS, 0.12),
]

# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentSpec:
    """How a component type ages, fails and announces itself in telemetry.

    `life_days` is the expected life under *nominal* stress. The generator
    accumulates damage daily at a rate scaled by stress sensitivities, so a
    unit in Ahmedabad that short-cycles reaches its threshold materially
    sooner than the same unit in Bengaluru - which is the signal the survival
    model has to learn.
    """

    key: str
    label: str
    life_days: float
    weibull_shape: float  # >1 = wear-out; 1 = random; <1 = infant mortality
    failure_modes: tuple[tuple[str, float], ...]  # (mode, probability)
    # Stress sensitivities (multiplier applied per unit of normalised stress)
    thermal_sens: float = 0.0  # degree-hours above design
    cycling_sens: float = 0.0  # starts per day
    load_sens: float = 0.0  # duty fraction
    runtime_sens: float = 1.0  # runtime hours (most components age on runtime)
    # Telemetry signature at full damage (damage_frac = 1.0)
    signature: dict[str, float] = field(default_factory=dict)
    pm_replaceable: bool = False  # reset on a PM visit (filters, belts)
    criticality: str = "B"


COMPONENTS: dict[str, ComponentSpec] = {
    "compressor": ComponentSpec(
        key="compressor", label="Compressor", life_days=4000, weibull_shape=2.8,
        failure_modes=(("mechanical_wear", 0.68), ("burnout", 0.32)),
        thermal_sens=0.9, cycling_sens=1.1, load_sens=0.7, runtime_sens=1.0,
        signature={"comp_current": 9.0, "vib_rms": 3.2, "cop": -0.38, "discharge_p": 160.0},
        criticality="A",
    ),
    "condenser_coil": ComponentSpec(
        key="condenser_coil", label="Condenser coil", life_days=1600, weibull_shape=2.2,
        failure_modes=(("fouling", 1.0),),
        thermal_sens=0.6, cycling_sens=0.1, load_sens=0.4, runtime_sens=0.8,
        # Fouling is the textbook signature: head pressure up, COP and subcooling move
        signature={"discharge_p": 520.0, "cop": -0.52, "subcool": 2.4, "comp_current": 3.1},
        criticality="B",
    ),
    "condenser_fan_motor": ComponentSpec(
        key="condenser_fan_motor", label="Condenser fan motor", life_days=2200, weibull_shape=2.4,
        failure_modes=(("bearing_failure", 0.62), ("winding_failure", 0.38)),
        thermal_sens=0.7, cycling_sens=0.8, load_sens=0.3, runtime_sens=1.0,
        signature={"vib_rms": 4.8, "vib_kurtosis": 3.6, "discharge_p": 180.0},
        criticality="B",
    ),
    "evap_blower_motor": ComponentSpec(
        key="evap_blower_motor", label="Evaporator blower motor", life_days=2400, weibull_shape=2.3,
        failure_modes=(("bearing_failure", 0.70), ("winding_failure", 0.30)),
        thermal_sens=0.2, cycling_sens=0.7, load_sens=0.4, runtime_sens=1.0,
        signature={"vib_rms": 4.2, "vib_kurtosis": 3.0, "supply_t": 1.1},
        criticality="B",
    ),
    "evaporator": ComponentSpec(
        key="evaporator", label="Evaporator", life_days=3200, weibull_shape=2.0,
        failure_modes=(("fouling", 0.75), ("tube_leak", 0.25)),
        thermal_sens=0.2, cycling_sens=0.1, load_sens=0.5, runtime_sens=0.7,
        signature={"suction_p": -90.0, "superheat": 2.8, "cop": -0.26},
        criticality="B",
    ),
    "cooling_coil": ComponentSpec(
        key="cooling_coil", label="Cooling coil", life_days=2600, weibull_shape=2.0,
        failure_modes=(("fouling", 0.82), ("tube_leak", 0.18)),
        thermal_sens=0.2, cycling_sens=0.1, load_sens=0.5, runtime_sens=0.7,
        signature={"supply_t": 2.2, "cop": -0.20},
        criticality="B",
    ),
    "eev": ComponentSpec(
        key="eev", label="Electronic expansion valve", life_days=2800, weibull_shape=1.9,
        failure_modes=(("valve_stuck", 0.58), ("sensor_drift", 0.42)),
        thermal_sens=0.3, cycling_sens=0.9, load_sens=0.2, runtime_sens=0.8,
        signature={"superheat": 6.5, "suction_p": -70.0, "cop": -0.22},
        criticality="B",
    ),
    "txv": ComponentSpec(
        key="txv", label="Thermostatic expansion valve", life_days=3000, weibull_shape=1.9,
        failure_modes=(("valve_stuck", 0.65), ("charge_loss", 0.35)),
        thermal_sens=0.3, cycling_sens=0.8, load_sens=0.2, runtime_sens=0.8,
        signature={"superheat": 7.0, "suction_p": -60.0},
        criticality="B",
    ),
    "vfd": ComponentSpec(
        key="vfd", label="Variable frequency drive", life_days=2600, weibull_shape=2.1,
        failure_modes=(("vfd_fault", 0.72), ("capacitor_degradation", 0.28)),
        # Power electronics age thermally (Arrhenius-ish), barely on runtime
        thermal_sens=1.4, cycling_sens=0.5, load_sens=0.6, runtime_sens=0.5,
        signature={"comp_current": 2.4, "current_imbalance": 3.2},
        criticality="A",
    ),
    "control_board": ComponentSpec(
        key="control_board", label="Control board", life_days=3400, weibull_shape=1.7,
        failure_modes=(("board_failure", 0.80), ("sensor_drift", 0.20)),
        thermal_sens=1.5, cycling_sens=0.3, load_sens=0.1, runtime_sens=0.3,
        signature={"current_imbalance": 1.6},
        criticality="A",
    ),
    "run_capacitor": ComponentSpec(
        key="run_capacitor", label="Run capacitor", life_days=1200, weibull_shape=2.6,
        failure_modes=(("capacitor_degradation", 1.0),),
        # Electrolytic capacitors are the classic thermal-ageing part
        thermal_sens=1.8, cycling_sens=1.2, load_sens=0.2, runtime_sens=0.6,
        signature={"current_imbalance": 5.8, "comp_current": 2.0},
        criticality="C",
    ),
    "contactor": ComponentSpec(
        key="contactor", label="Contactor", life_days=1000, weibull_shape=3.0,
        failure_modes=(("contactor_pitting", 1.0),),
        # Contactors wear per *operation*, so cycling dominates
        thermal_sens=0.4, cycling_sens=2.4, load_sens=0.1, runtime_sens=0.3,
        signature={"current_imbalance": 3.4},
        criticality="C",
    ),
    "refrigerant_circuit": ComponentSpec(
        key="refrigerant_circuit", label="Refrigerant circuit", life_days=1800, weibull_shape=1.4,
        failure_modes=(("refrigerant_leak", 1.0),),
        thermal_sens=0.6, cycling_sens=0.7, load_sens=0.2, runtime_sens=0.6,
        # Leak: suction pressure falls, superheat rises, charge falls
        signature={"suction_p": -210.0, "superheat": 9.5, "charge_frac": -0.30, "cop": -0.30},
        criticality="B",
    ),
    "fan_bearing": ComponentSpec(
        key="fan_bearing", label="Fan bearing", life_days=1500, weibull_shape=2.5,
        failure_modes=(("bearing_failure", 1.0),),
        thermal_sens=0.5, cycling_sens=0.9, load_sens=0.3, runtime_sens=1.0,
        signature={"vib_rms": 5.5, "vib_kurtosis": 4.4},
        criticality="C",
    ),
    "damper_actuator": ComponentSpec(
        key="damper_actuator", label="Damper actuator", life_days=2000, weibull_shape=2.0,
        failure_modes=(("actuator_stuck", 1.0),),
        thermal_sens=0.3, cycling_sens=1.6, load_sens=0.1, runtime_sens=0.5,
        signature={"supply_t": 1.4},
        criticality="C",
    ),
    "belt": ComponentSpec(
        key="belt", label="Drive belt", life_days=520, weibull_shape=2.2,
        failure_modes=(("belt_wear", 1.0),),
        thermal_sens=0.4, cycling_sens=0.6, load_sens=0.8, runtime_sens=1.0,
        signature={"vib_rms": 2.2, "supply_t": 0.7},
        pm_replaceable=True, criticality="C",
    ),
    "filter": ComponentSpec(
        key="filter", label="Air filter", life_days=130, weibull_shape=3.4,
        failure_modes=(("filter_clogged", 1.0),),
        thermal_sens=0.1, cycling_sens=0.1, load_sens=0.9, runtime_sens=1.0,
        signature={"supply_t": 1.6, "cop": -0.12},
        pm_replaceable=True, criticality="C",
    ),
}

# --------------------------------------------------------------------------
# Parts, BOM, repair kits
# --------------------------------------------------------------------------

SIZE_CLASSES = ("S", "M", "L")

# Component types whose part number depends on machine size.
SIZED_COMPONENTS = {
    "compressor", "condenser_coil", "condenser_fan_motor", "evap_blower_motor",
    "evaporator", "cooling_coil", "vfd", "control_board", "belt", "filter",
}


@dataclass(frozen=True)
class PartSpec:
    part_no: str
    description: str
    unit_cost: float
    criticality: str
    shelf_life_days: int | None
    lead_time_mean_d: float
    lead_time_sd_d: float
    moq: int
    expedite_cost_mult: float


def _sized(prefix: str, size: str) -> str:
    return f"{prefix}-{size}"


# Base cost by component type for the S size; M and L scale up.
_BASE_COST = {
    "compressor": 68_000, "condenser_coil": 22_000, "condenser_fan_motor": 9_400,
    "evap_blower_motor": 11_200, "evaporator": 26_000, "cooling_coil": 18_500,
    "eev": 7_800, "txv": 4_200, "vfd": 42_000, "control_board": 19_500,
    "run_capacitor": 1_250, "contactor": 1_480, "refrigerant_circuit": 0,
    "fan_bearing": 2_300, "damper_actuator": 6_400, "belt": 850, "filter": 620,
}
_SIZE_COST_MULT = {"S": 1.0, "M": 1.55, "L": 2.45}

# Lead time by component type: long-lead capital parts vs shelf commodities.
_LEAD = {
    "compressor": (21.0, 4.0, 1), "condenser_coil": (18.0, 3.5, 1),
    "condenser_fan_motor": (9.0, 2.0, 1), "evap_blower_motor": (10.0, 2.5, 1),
    "evaporator": (24.0, 5.0, 1), "cooling_coil": (20.0, 4.0, 1),
    "eev": (12.0, 3.0, 2), "txv": (8.0, 2.0, 2), "vfd": (28.0, 6.0, 1),
    "control_board": (16.0, 4.0, 1), "run_capacitor": (5.0, 1.2, 10),
    "contactor": (6.0, 1.5, 5), "fan_bearing": (7.0, 1.8, 4),
    "damper_actuator": (11.0, 2.5, 2), "belt": (4.0, 1.0, 5), "filter": (3.0, 0.8, 20),
}


def build_parts_catalog() -> list[PartSpec]:
    """Every part number in the world, with cost, lead time and MOQ."""
    parts: list[PartSpec] = []
    for ckey, spec in COMPONENTS.items():
        if ckey == "refrigerant_circuit":
            continue  # repaired with consumables, not a replaceable part
        lt_mean, lt_sd, moq = _LEAD[ckey]
        if ckey in SIZED_COMPONENTS:
            for size in SIZE_CLASSES:
                parts.append(
                    PartSpec(
                        part_no=_sized(_PART_PREFIX[ckey], size),
                        description=f"{spec.label}, size {size}",
                        unit_cost=round(_BASE_COST[ckey] * _SIZE_COST_MULT[size], 2),
                        criticality=spec.criticality,
                        shelf_life_days=None if ckey != "filter" else 1095,
                        lead_time_mean_d=lt_mean,
                        lead_time_sd_d=lt_sd,
                        moq=moq,
                        expedite_cost_mult=2.4 if spec.criticality == "A" else 1.9,
                    )
                )
        else:
            parts.append(
                PartSpec(
                    part_no=_PART_PREFIX[ckey],
                    description=spec.label,
                    unit_cost=float(_BASE_COST[ckey]),
                    criticality=spec.criticality,
                    shelf_life_days=None,
                    lead_time_mean_d=lt_mean,
                    lead_time_sd_d=lt_sd,
                    moq=moq,
                    expedite_cost_mult=2.0,
                )
            )
    parts.extend(CONSUMABLES)
    return parts


_PART_PREFIX = {
    "compressor": "CMP", "condenser_coil": "CDC", "condenser_fan_motor": "CFM",
    "evap_blower_motor": "EBM", "evaporator": "EVP", "cooling_coil": "CLC",
    "eev": "EEV-STD", "txv": "TXV-STD", "vfd": "VFD", "control_board": "CTB",
    "run_capacitor": "CAP-45-5", "contactor": "CNT-3P-40A",
    "fan_bearing": "BRG-6205", "damper_actuator": "DMP-ACT-24V",
    "belt": "BELT", "filter": "FLT",
}

# Size-independent consumables used across repair kits.
CONSUMABLES: list[PartSpec] = [
    PartSpec("REF-R32-KG", "Refrigerant R32 (kg)", 1_120, "B", 1825, 7.0, 2.0, 10, 1.8),
    PartSpec("REF-R410A-KG", "Refrigerant R410A (kg)", 980, "B", 1825, 7.0, 2.0, 10, 1.8),
    PartSpec("FD-34", "Filter-drier, 3/4 in", 2_150, "B", 1095, 6.0, 1.5, 5, 1.9),
    PartSpec("BRZ-KIT", "Brazing kit / service valve core", 1_650, "C", 730, 4.0, 1.0, 5, 1.6),
    PartSpec("OIL-POE-L", "POE compressor oil (L)", 1_380, "C", 1095, 6.0, 1.5, 6, 1.6),
    PartSpec("GSKT-SET", "Gasket set", 740, "C", 1460, 5.0, 1.2, 10, 1.5),
]

# ---------------------------------------------------------------------------
# Repair kits: (component_type, failure_mode) -> [(part_key, qty, p_used)]
#
# `part_key` is either a literal part number or "@self" meaning "the part for
# this component at this asset's size class". p_used is the probability the
# part is actually consumed on that repair - this is what makes the demand
# calculation probabilistic rather than a naive BOM explosion.
# ---------------------------------------------------------------------------

REPAIR_KITS: dict[tuple[str, str], list[tuple[str, float, float]]] = {
    ("compressor", "mechanical_wear"): [
        ("@self", 1, 1.00), ("FD-34", 1, 0.95), ("@refrigerant", 4.5, 1.00),
        ("OIL-POE-L", 2, 0.85), ("CNT-3P-40A", 1, 0.40), ("GSKT-SET", 1, 0.55),
    ],
    ("compressor", "burnout"): [
        ("@self", 1, 1.00), ("FD-34", 2, 1.00), ("@refrigerant", 5.5, 1.00),
        ("OIL-POE-L", 3, 1.00), ("CNT-3P-40A", 1, 0.75), ("CAP-45-5", 1, 0.50),
    ],
    ("condenser_coil", "fouling"): [
        ("@self", 1, 0.35), ("BRZ-KIT", 1, 0.30), ("@refrigerant", 2.0, 0.35),
    ],
    ("condenser_fan_motor", "bearing_failure"): [
        ("@self", 1, 1.00), ("CAP-45-5", 1, 0.60), ("BRG-6205", 2, 0.25),
    ],
    ("condenser_fan_motor", "winding_failure"): [
        ("@self", 1, 1.00), ("CAP-45-5", 1, 0.70), ("CNT-3P-40A", 1, 0.30),
    ],
    ("evap_blower_motor", "bearing_failure"): [
        ("@self", 1, 1.00), ("BRG-6205", 2, 0.30), ("BELT", 1, 0.45),
    ],
    ("evap_blower_motor", "winding_failure"): [
        ("@self", 1, 1.00), ("CAP-45-5", 1, 0.55),
    ],
    ("evaporator", "fouling"): [("@self", 1, 0.25), ("GSKT-SET", 1, 0.40)],
    ("evaporator", "tube_leak"): [
        ("@self", 1, 0.80), ("BRZ-KIT", 1, 1.00), ("@refrigerant", 3.5, 1.00),
    ],
    ("cooling_coil", "fouling"): [("@self", 1, 0.22), ("GSKT-SET", 1, 0.35)],
    ("cooling_coil", "tube_leak"): [("@self", 1, 0.75), ("BRZ-KIT", 1, 1.00)],
    ("eev", "valve_stuck"): [("@self", 1, 1.00), ("@refrigerant", 1.5, 0.70), ("FD-34", 1, 0.45)],
    ("eev", "sensor_drift"): [("@self", 1, 0.55)],
    ("txv", "valve_stuck"): [("@self", 1, 1.00), ("@refrigerant", 1.5, 0.75), ("BRZ-KIT", 1, 0.60)],
    ("txv", "charge_loss"): [("@refrigerant", 2.5, 1.00), ("BRZ-KIT", 1, 0.80), ("@self", 1, 0.30)],
    ("vfd", "vfd_fault"): [("@self", 1, 1.00), ("CNT-3P-40A", 1, 0.25)],
    ("vfd", "capacitor_degradation"): [("CAP-45-5", 2, 1.00), ("@self", 1, 0.35)],
    ("control_board", "board_failure"): [("@self", 1, 1.00), ("CNT-3P-40A", 1, 0.20)],
    ("control_board", "sensor_drift"): [("@self", 1, 0.40)],
    ("run_capacitor", "capacitor_degradation"): [("CAP-45-5", 1, 1.00), ("CNT-3P-40A", 1, 0.25)],
    ("contactor", "contactor_pitting"): [("CNT-3P-40A", 1, 1.00), ("CAP-45-5", 1, 0.15)],
    ("refrigerant_circuit", "refrigerant_leak"): [
        ("@refrigerant", 2.0, 1.00), ("BRZ-KIT", 1, 0.50), ("FD-34", 1, 0.35),
    ],
    ("fan_bearing", "bearing_failure"): [("BRG-6205", 2, 1.00), ("BELT", 1, 0.30)],
    ("damper_actuator", "actuator_stuck"): [("DMP-ACT-24V", 1, 1.00)],
    ("belt", "belt_wear"): [("BELT", 1, 1.00)],
    ("filter", "filter_clogged"): [("FLT", 1, 1.00)],
}

# Parts cheap enough and probable enough to be worth putting on the van
# speculatively - the first-time-fix lever.
VAN_KIT_EXTRA_MAX_COST = 2_500.0
VAN_KIT_MIN_P_USED = 0.20

# --------------------------------------------------------------------------
# Technician note templates (a fourth "sensor" - an LLM maps these to codes)
# --------------------------------------------------------------------------

NOTE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "mechanical_wear": (
        "Compressor noisy under load, current draw high. Replaced compressor and filter-drier.",
        "Found scored compressor, elevated discharge temp. Swapped unit, recharged.",
        "High amps and rattle at startup. Compressor replacement, system evacuated and recharged.",
    ),
    "burnout": (
        "Compressor burnout, acid test positive. Replaced compressor, double filter-drier, flushed.",
        "Windings shorted to ground. Compressor and contactor replaced, oil changed.",
    ),
    "fouling": (
        "Condenser coil heavily fouled, head pressure high. Chemical clean performed.",
        "Coil blocked with debris, approach temp out of spec. Cleaned and rinsed.",
        "Fouling noted at PM, partial clean only - recommend follow-up visit.",
    ),
    "bearing_failure": (
        "Fan bearing noise and vibration. Replaced motor assembly.",
        "Rough bearing on blower, high vibration reading. Motor and bearings replaced.",
    ),
    "winding_failure": (
        "Motor tripping on overload, windings open. Replaced motor and capacitor.",
    ),
    "refrigerant_leak": (
        "Low suction, high superheat. Found leak at service valve, repaired and recharged 2 kg.",
        "Charge low on inspection. Leak traced to brazed joint, repaired and topped up.",
    ),
    "valve_stuck": (
        "Superheat erratic, EEV not modulating. Replaced valve and recovered charge.",
        "Expansion valve stuck partially closed. Valve replaced, superheat reset to spec.",
    ),
    "sensor_drift": ("Suction temp sensor reading offset. Sensor replaced and controller recalibrated.",),
    "capacitor_degradation": (
        "Start current imbalance, capacitor out of tolerance. Run capacitor replaced.",
        "Capacitor bulged, measured 28 uF against 45 uF rated. Replaced.",
    ),
    "contactor_pitting": (
        "Contactor points pitted and welding. Replaced contactor.",
        "Chatter on contactor pull-in, contacts burnt. Replacement fitted.",
    ),
    "vfd_fault": ("Drive showing overcurrent fault code, would not reset. VFD replaced.",),
    "board_failure": ("Controller unresponsive, no display. Control board replaced and parameters reloaded.",),
    "tube_leak": ("Coil leaking at return bend. Brazed repair, pressure tested and recharged.",),
    "actuator_stuck": ("Economiser damper not driving. Actuator replaced and stroked.",),
    "belt_wear": ("Belt glazed and cracked, slipping under load. Replaced and re-tensioned.",),
    "filter_clogged": ("Filters loaded, high pressure drop across bank. Full filter change.",),
}

PM_NOTE_TEMPLATES: tuple[str, ...] = (
    "Routine PM. Filters changed, coils inspected, charge and superheat checked - within spec.",
    "Scheduled PM. Belt tension adjusted, filters replaced, electricals torqued.",
    "Quarterly PM completed. Minor coil fouling noted and cleaned; no parts required beyond filters.",
    "PM visit. Filter bank replaced, condensate drain cleared, controls verified.",
)
