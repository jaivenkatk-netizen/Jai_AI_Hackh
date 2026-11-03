"""Asset twin: a steady-state vapour-compression cycle per machine.

What it is for, concretely. The survival model sees raw telemetry, so it cannot
separate "this compressor is dying" from "it was 42 C yesterday" - both raise
head pressure and current. The twin computes what a *healthy* machine would
read under the same ambient, load and setpoints, and the residual
(observed - healthy) isolates degradation from operating conditions.

That residual is the principled fix for a failure found in the Day-1 backtest:
three heavily-degraded compressors were predicted at 580-3565 days because
their only in-distribution signal, `n_past_repairs`, was 0. Their telemetry was
plainly abnormal (43 A against a fleet-typical 26-30 A) but the model could not
tell abnormal-for-conditions from hot-day-normal.

The model
---------
Four components, solved simultaneously for the two unknown saturation
temperatures:

  compressor   10-coefficient AHRI 540-style map -> mass flow and power, with a
               clearance-volume volumetric efficiency so pressure ratio
               correctly suppresses throughput
  condenser    lumped UA: Q = UA_c * (T_cond - T_ambient)
  evaporator   lumped UA: Q = UA_e * (T_air_in - T_evap)
  TXV / EEV    holds a superheat setpoint

Closing the loop on the two energy balances gives T_evap and T_cond, and from
those every gauge reading a technician would see.

Faults enter as *physical parameter changes*, not as offsets bolted onto the
output - which is what makes the residual meaningful rather than circular:

  condenser fouling   UA_c falls            -> head pressure up, COP down
  refrigerant leak    charge falls          -> suction down, superheat up
  compressor wear     isentropic eff. falls -> current up, capacity down
  fan bearing wear    air flow falls        -> UA_c falls (same path as fouling)

Refrigerant properties come from CoolProp. If it is unavailable the module
falls back to fitted correlations in sim/refrig_props.py, because a demo must
not hard-fail on an optional binary dependency.

Run:
    python -m sim.asset_twin --self-test
    python -m sim.asset_twin --sweep-ambient
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

try:  # CoolProp is the reference implementation; the fallback keeps the demo alive
    import CoolProp.CoolProp as CP

    HAVE_COOLPROP = True
except Exception:  # pragma: no cover - exercised only on machines without the wheel
    from sim import refrig_props as CP  # type: ignore

    HAVE_COOLPROP = False

# Physical constants
CP_AIR = 1005.0  # J/kg.K
R_SQRT3 = math.sqrt(3.0)


# --------------------------------------------------------------------------
# Refrigerant property helpers
# --------------------------------------------------------------------------


def p_sat(fluid: str, t_c: float) -> float:
    """Saturation pressure [kPa] at temperature [C]."""
    return float(CP.PropsSI("P", "T", t_c + 273.15, "Q", 0, fluid)) / 1000.0


def t_sat(fluid: str, p_kpa: float) -> float:
    """Saturation temperature [C] at pressure [kPa]."""
    return float(CP.PropsSI("T", "P", p_kpa * 1000.0, "Q", 0, fluid)) - 273.15


def t_crit(fluid: str) -> float:
    """Critical temperature [C].

    The condensing-temperature search must stay below this: above the critical
    point there is no saturation line to evaluate, and CoolProp rightly refuses
    ("Temperature to QT_flash may not be above the numerical critical point").
    Physically a machine trips on high pressure long before here - R32 crits at
    78.1 C, R410A at 71.3 C - so the cap is a real operating bound, not just a
    numerical guard.
    """
    try:
        return float(CP.PropsSI("Tcrit", fluid)) - 273.15
    except Exception:
        return {"R32": 78.1, "R410A": 71.3}.get(fluid, 70.0)


def h_superheated(fluid: str, t_c: float, p_kpa: float) -> float:
    """Specific enthalpy [J/kg] of superheated vapour."""
    return float(CP.PropsSI("H", "T", t_c + 273.15, "P", p_kpa * 1000.0, fluid))


def rho_superheated(fluid: str, t_c: float, p_kpa: float) -> float:
    return float(CP.PropsSI("D", "T", t_c + 273.15, "P", p_kpa * 1000.0, fluid))


def s_superheated(fluid: str, t_c: float, p_kpa: float) -> float:
    return float(CP.PropsSI("S", "T", t_c + 273.15, "P", p_kpa * 1000.0, fluid))


def h_from_ps(fluid: str, p_kpa: float, s: float) -> float:
    return float(CP.PropsSI("H", "P", p_kpa * 1000.0, "S", s, fluid))


def h_liquid(fluid: str, t_c: float) -> float:
    """Saturated liquid enthalpy [J/kg]."""
    return float(CP.PropsSI("H", "T", t_c + 273.15, "Q", 0, fluid))


def h_subcooled(fluid: str, t_sat_c: float, subcool_k: float, p_kpa: float) -> float:
    """Enthalpy of liquid subcooled below its saturation temperature."""
    t = max(t_sat_c - subcool_k, -60.0)
    try:
        return float(CP.PropsSI("H", "T", t + 273.15, "P", p_kpa * 1000.0, fluid))
    except Exception:
        # Near the saturation line CoolProp can refuse a P,T pair; fall back to
        # cp*dT off the saturated liquid, which is accurate for small subcooling.
        return h_liquid(fluid, t_sat_c) - 1500.0 * subcool_k


# --------------------------------------------------------------------------
# Machine definition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TwinSpec:
    """Design parameters of one machine. Faults are expressed as multipliers."""

    refrigerant: str = "R32"
    capacity_kw: float = 280.0

    # Compressor
    # Calibrated so the model delivers nameplate capacity at 35 C ambient:
    # the first guess of 0.065 produced 401 kW from a 280 kW machine. At this
    # value the twin reports 280 kW at COP 5.0 and 0.70 kW/ton, which sits in
    # the 0.55-0.75 band for a good chiller - so the absolute numbers are
    # defensible, not just the fault directions.
    displacement_m3_s: float = 0.03653  # swept volume rate at full speed
    clearance: float = 0.045  # clearance-volume fraction
    poly_n: float = 1.12  # re-expansion polytropic index
    isentropic_eff: float = 0.72
    motor_eff: float = 0.92
    voltage: float = 415.0
    power_factor: float = 0.86

    # Heat exchangers, as lumped conductances [W/K]
    ua_cond: float = 26_000.0
    ua_evap: float = 30_000.0

    # Control
    superheat_sp: float = 5.5
    subcool_sp: float = 4.5
    air_in_c: float = 24.0  # return air / chilled-water return

    # ---- fault state: all 1.0 (or 0.0) means healthy -------------------
    foul_frac: float = 0.0  # condenser fouling          -> UA_cond down
    wear_frac: float = 0.0  # compressor wear            -> isentropic eff down
    charge_frac: float = 1.0  # refrigerant charge        -> starved evaporator
    airflow_frac: float = 1.0  # condenser fan airflow    -> UA_cond down
    evap_foul_frac: float = 0.0  # evaporator fouling     -> UA_evap down

    # ---- effective (degraded) parameters -------------------------------
    @property
    def ua_cond_eff(self) -> float:
        # Fouling and reduced airflow both attack the same conductance. Airflow
        # enters to the 0.6 power, the usual air-side exponent for forced
        # convection over a finned coil.
        return self.ua_cond * (1.0 - 0.75 * self.foul_frac) * (self.airflow_frac**0.6)

    # Evaporator dry-out exponent. Undercharge leaves part of the coil carrying
    # superheated vapour instead of boiling refrigerant, and the vapour-side
    # heat transfer coefficient is one to two orders of magnitude worse than
    # two-phase - so effective conductance collapses far faster than the wetted
    # fraction itself. A sqrt law (the first guess) gave only a 2.8% suction
    # drop at 28% undercharge; the field figure is 15-25%. Calibrated against
    # that: this exponent yields ~16%.
    CHARGE_DRYOUT_EXPONENT = 2.2

    @property
    def ua_evap_eff(self) -> float:
        return (
            self.ua_evap
            * (1.0 - 0.6 * self.evap_foul_frac)
            * (min(self.charge_frac, 1.0) ** self.CHARGE_DRYOUT_EXPONENT)
        )

    @property
    def isentropic_eff_eff(self) -> float:
        return max(self.isentropic_eff * (1.0 - 0.35 * self.wear_frac), 0.25)

    @property
    def superheat_eff(self) -> float:
        # An undercharged circuit cannot hold the valve's setpoint: the TXV runs
        # wide open and superheat climbs steeply. This is the signature a
        # technician reads as "low on gas".
        deficit = max(0.0, 1.0 - self.charge_frac)
        return self.superheat_sp + 45.0 * deficit

    @property
    def subcool_eff(self) -> float:
        # Fouling raises subcooling (liquid backs up); a leak destroys it.
        deficit = max(0.0, 1.0 - self.charge_frac)
        return max(self.subcool_sp + 3.0 * self.foul_frac - 9.0 * deficit, 0.2)


@dataclass(frozen=True)
class TwinState:
    """What the gauges and the ammeter would read."""

    t_evap_c: float
    t_cond_c: float
    suction_p_kpa: float
    discharge_p_kpa: float
    superheat_k: float
    subcool_k: float
    mass_flow_kg_s: float
    q_evap_w: float
    q_cond_w: float
    w_comp_w: float
    cop: float
    comp_current_a: float
    pressure_ratio: float
    converged: bool


# --------------------------------------------------------------------------
# The cycle solve
# --------------------------------------------------------------------------


def _volumetric_eff(spec: TwinSpec, pressure_ratio: float) -> float:
    """Clearance-volume volumetric efficiency.

    eta_v = 1 - C * (PR^(1/n) - 1). This is what makes a fouled machine lose
    capacity rather than merely run hotter: a higher pressure ratio re-expands
    more clearance gas, so less fresh charge is drawn in each stroke.
    """
    pr = max(pressure_ratio, 1.01)
    return float(np.clip(1.0 - spec.clearance * (pr ** (1.0 / spec.poly_n) - 1.0), 0.25, 1.0))


def _compressor(spec: TwinSpec, t_evap: float, t_cond: float, speed_frac: float) -> tuple[float, float, float]:
    """Return (mass flow [kg/s], compressor power [W], discharge enthalpy)."""
    f = spec.refrigerant
    p_suc = p_sat(f, t_evap)
    p_dis = p_sat(f, t_cond)
    pr = p_dis / max(p_suc, 1e-6)

    t_suc = t_evap + spec.superheat_eff
    rho = rho_superheated(f, t_suc, p_suc)
    h_suc = h_superheated(f, t_suc, p_suc)
    s_suc = s_superheated(f, t_suc, p_suc)

    eta_v = _volumetric_eff(spec, pr)
    m_dot = spec.displacement_m3_s * speed_frac * rho * eta_v

    h_dis_isen = h_from_ps(f, p_dis, s_suc)
    h_dis = h_suc + (h_dis_isen - h_suc) / spec.isentropic_eff_eff
    w_comp = m_dot * (h_dis - h_suc)
    return m_dot, w_comp, h_dis


def _residuals(spec: TwinSpec, t_evap: float, t_cond: float, ambient_c: float, speed_frac: float):
    """Energy-balance residuals for the two heat exchangers."""
    f = spec.refrigerant
    m_dot, w_comp, h_dis = _compressor(spec, t_evap, t_cond, speed_frac)

    p_dis = p_sat(f, t_cond)
    h_liq_out = h_subcooled(f, t_cond, spec.subcool_eff, p_dis)

    # Condenser: refrigerant-side rejection vs air-side capability
    q_cond_ref = m_dot * (h_dis - h_liq_out)
    q_cond_air = spec.ua_cond_eff * (t_cond - ambient_c)

    # Evaporator: refrigerant-side absorption vs air-side capability.
    # The valve is isenthalpic, so liquid enters the evaporator at h_liq_out.
    t_suc = t_evap + spec.superheat_eff
    h_suc = h_superheated(f, t_suc, p_sat(f, t_evap))
    q_evap_ref = m_dot * (h_suc - h_liq_out)
    q_evap_air = spec.ua_evap_eff * (spec.air_in_c - t_evap)

    return q_cond_ref - q_cond_air, q_evap_ref - q_evap_air


def solve(spec: TwinSpec, ambient_c: float, load_frac: float = 1.0) -> TwinState:
    """Find the steady operating point for these conditions.

    Nested one-dimensional bracketing rather than a 2-D Newton solve: each
    residual is monotonic in its own saturation temperature, so brentq is
    both unconditionally convergent here and far more robust than fsolve,
    which wandered into CoolProp's two-phase region and raised.
    """
    speed = float(np.clip(load_frac, 0.15, 1.0))

    def evap_balance(t_evap: float, t_cond: float) -> float:
        return _residuals(spec, t_evap, t_cond, ambient_c, speed)[1]

    def cond_balance(t_cond: float) -> float:
        # Inner solve: the evaporator temperature consistent with this T_cond
        lo, hi = -25.0, spec.air_in_c - 0.5
        try:
            f_lo, f_hi = evap_balance(lo, t_cond), evap_balance(hi, t_cond)
        except Exception:
            return 1e9
        if f_lo * f_hi > 0:
            return 1e9 if f_hi > 0 else -1e9
        t_evap = brentq(evap_balance, lo, hi, args=(t_cond,), xtol=1e-3, maxiter=60)
        return _residuals(spec, t_evap, t_cond, ambient_c, speed)[0]

    # Stay clear of the critical point: there is no saturation line above it.
    # 3 K of margin keeps the property calls away from the numerical cliff.
    t_max = t_crit(spec.refrigerant) - 3.0
    lo, hi = ambient_c + 1.0, min(ambient_c + 55.0, t_max)
    converged = True
    if hi <= lo:
        # Ambient at or above the critical temperature - the machine cannot
        # reject heat at all. Report the bound and flag non-convergence.
        hi = lo + 0.5
        converged = False
    try:
        t_cond = brentq(cond_balance, lo, hi, xtol=1e-3, maxiter=60)
    except Exception:
        # Degraded machines can genuinely have no steady solution in range -
        # physically, the unit would trip on high pressure. Report the bound
        # rather than raising, and flag it.
        t_cond, converged = hi, False

    lo_e, hi_e = -25.0, spec.air_in_c - 0.5
    try:
        t_evap = brentq(evap_balance, lo_e, hi_e, args=(t_cond,), xtol=1e-3, maxiter=60)
    except Exception:
        t_evap, converged = lo_e, False

    f = spec.refrigerant
    m_dot, w_comp, h_dis = _compressor(spec, t_evap, t_cond, speed)
    p_suc, p_dis = p_sat(f, t_evap), p_sat(f, t_cond)
    h_liq_out = h_subcooled(f, t_cond, spec.subcool_eff, p_dis)
    h_suc = h_superheated(f, t_evap + spec.superheat_eff, p_suc)

    q_evap = m_dot * (h_suc - h_liq_out)
    q_cond = m_dot * (h_dis - h_liq_out)
    cop = q_evap / w_comp if w_comp > 1.0 else 0.0
    # Three-phase motor current from shaft power
    current = w_comp / (R_SQRT3 * spec.voltage * spec.power_factor * spec.motor_eff)

    return TwinState(
        t_evap_c=t_evap,
        t_cond_c=t_cond,
        suction_p_kpa=p_suc,
        discharge_p_kpa=p_dis,
        superheat_k=spec.superheat_eff,
        subcool_k=spec.subcool_eff,
        mass_flow_kg_s=m_dot,
        q_evap_w=q_evap,
        q_cond_w=q_cond,
        w_comp_w=w_comp,
        cop=cop,
        comp_current_a=current,
        pressure_ratio=p_dis / max(p_suc, 1e-6),
        converged=converged,
    )


# --------------------------------------------------------------------------
# Residual features - the reason the twin exists
# --------------------------------------------------------------------------

RESIDUAL_SIGNALS = ("discharge_p", "suction_p", "superheat", "subcool", "comp_current", "cop")


def healthy_baseline(spec: TwinSpec, ambient_c: float, load_frac: float) -> dict[str, float]:
    """What a *healthy* machine of this design would read right now."""
    clean = replace(
        spec, foul_frac=0.0, wear_frac=0.0, charge_frac=1.0, airflow_frac=1.0, evap_foul_frac=0.0
    )
    s = solve(clean, ambient_c, load_frac)
    return {
        "discharge_p": s.discharge_p_kpa,
        "suction_p": s.suction_p_kpa,
        "superheat": s.superheat_k,
        "subcool": s.subcool_k,
        "comp_current": s.comp_current_a,
        "cop": s.cop,
    }


def residuals(
    observed: dict[str, float], spec: TwinSpec, ambient_c: float, load_frac: float
) -> dict[str, float]:
    """observed - healthy, per signal.

    This is the feature the survival model actually wants: it is near zero on a
    healthy machine whatever the weather, and grows with degradation. Signals
    absent from `observed` (an air-only AHU has no refrigerant circuit) are
    skipped rather than imputed.
    """
    base = healthy_baseline(spec, ambient_c, load_frac)
    out: dict[str, float] = {}
    for k, b in base.items():
        v = observed.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        out[f"{k}_resid"] = float(v) - float(b)
        # Relative form too: a 200 kPa excess means different things on a
        # 2400 kPa head and a 900 kPa suction.
        if abs(b) > 1e-6:
            out[f"{k}_resid_rel"] = (float(v) - float(b)) / abs(float(b))
    return out


def spec_for_asset(
    asset_type: str, capacity_kw: float, refrigerant: str, air_in_c: float = 24.0
) -> TwinSpec:
    """Scale the design parameters to one machine from the fleet table."""
    # Displacement and conductances scale with rated capacity; the reference
    # design above is a ~280 kW chiller.
    k = max(capacity_kw, 5.0) / 280.0
    if refrigerant not in ("R32", "R410A"):
        refrigerant = "R32"  # air-only machines get a nominal circuit
    return TwinSpec(
        refrigerant=refrigerant,
        capacity_kw=capacity_kw,
        displacement_m3_s=0.03653 * k,
        ua_cond=26_000.0 * k,
        ua_evap=30_000.0 * k,
        air_in_c=air_in_c,
        # Rooftop and split units run hotter-condensing than a water chiller
        superheat_sp=6.0 if asset_type in ("rtu", "split_heat_pump", "vrf") else 5.5,
    )


# --------------------------------------------------------------------------
# Self-test: the fault directions must match the textbook
# --------------------------------------------------------------------------


def self_test() -> bool:
    """Assert each injected fault moves the gauges the way the textbook says.

    This is the test that makes the twin trustworthy as a baseline. If a fault
    moved a reading the wrong way, every residual built on it would be
    misleading, and the survival model would learn the inverse of the truth.
    """
    spec = TwinSpec()
    amb, load = 35.0, 1.0
    base = solve(spec, amb, load)
    print(f"  healthy @ {amb:.0f} C: suction {base.suction_p_kpa:,.0f} kPa, "
          f"discharge {base.discharge_p_kpa:,.0f} kPa, SH {base.superheat_k:.1f} K, "
          f"current {base.comp_current_a:.1f} A, COP {base.cop:.2f}")

    checks: list[tuple[str, bool, str]] = []

    foul = solve(replace(spec, foul_frac=0.8), amb, load)
    checks += [
        ("fouling raises head pressure", foul.discharge_p_kpa > base.discharge_p_kpa * 1.05,
         f"{base.discharge_p_kpa:,.0f} -> {foul.discharge_p_kpa:,.0f} kPa"),
        ("fouling lowers COP", foul.cop < base.cop * 0.95,
         f"{base.cop:.2f} -> {foul.cop:.2f}"),
        ("fouling raises current", foul.comp_current_a > base.comp_current_a,
         f"{base.comp_current_a:.1f} -> {foul.comp_current_a:.1f} A"),
    ]

    leak = solve(replace(spec, charge_frac=0.72), amb, load)
    checks += [
        # Magnitude matters: the field drop for a ~28% undercharge is 15-25%,
        # and a sign-only assertion passed happily when the model gave 2.8%.
        ("leak drops suction 10-30% (field: 15-25%)",
         0.10 <= 1 - leak.suction_p_kpa / base.suction_p_kpa <= 0.30,
         f"{base.suction_p_kpa:,.0f} -> {leak.suction_p_kpa:,.0f} kPa "
         f"({(1 - leak.suction_p_kpa / base.suction_p_kpa) * 100:.1f}%)"),
        ("leak raises superheat", leak.superheat_k > base.superheat_k + 3,
         f"{base.superheat_k:.1f} -> {leak.superheat_k:.1f} K"),
        ("leak lowers capacity", leak.q_evap_w < base.q_evap_w * 0.95,
         f"{base.q_evap_w / 1000:.0f} -> {leak.q_evap_w / 1000:.0f} kW"),
    ]

    wear = solve(replace(spec, wear_frac=0.8), amb, load)
    checks += [
        ("wear lowers COP", wear.cop < base.cop * 0.95, f"{base.cop:.2f} -> {wear.cop:.2f}"),
        ("wear lowers capacity", wear.q_evap_w <= base.q_evap_w,
         f"{base.q_evap_w / 1000:.0f} -> {wear.q_evap_w / 1000:.0f} kW"),
    ]

    hot = solve(spec, 45.0, load)
    checks += [
        ("hotter ambient raises head pressure", hot.discharge_p_kpa > base.discharge_p_kpa,
         f"{base.discharge_p_kpa:,.0f} -> {hot.discharge_p_kpa:,.0f} kPa"),
        ("hotter ambient lowers COP", hot.cop < base.cop, f"{base.cop:.2f} -> {hot.cop:.2f}"),
    ]

    checks.append(("healthy COP is plausible (2-7)", 2.0 < base.cop < 7.0, f"{base.cop:.2f}"))
    checks.append(
        ("healthy pressure ratio is plausible (1.5-6)", 1.5 < base.pressure_ratio < 6.0,
         f"{base.pressure_ratio:.2f}")
    )

    print()
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:42s} {detail}")
        ok &= passed
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Steady-state vapour-compression twin")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sweep-ambient", action="store_true")
    args = ap.parse_args()

    print(f"refrigerant properties: {'CoolProp ' + CP.get_global_param_string('version') if HAVE_COOLPROP else 'fitted fallback'}\n")

    if args.sweep_ambient:
        spec = TwinSpec()
        print("  amb C | suction | discharge | current |  COP  | capacity kW")
        for amb in (20, 25, 30, 35, 40, 45):
            s = solve(spec, float(amb), 1.0)
            print(f"  {amb:5.0f} | {s.suction_p_kpa:7.0f} | {s.discharge_p_kpa:9.0f} | "
                  f"{s.comp_current_a:7.1f} | {s.cop:5.2f} | {s.q_evap_w / 1000:11.0f}")
        return

    ok = self_test()
    print(f"\nSELF-TEST: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
