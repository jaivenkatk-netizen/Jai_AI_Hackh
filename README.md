# ServiceSense - A01: HVAC maintenance to inventory

One pipeline that turns HVAC telemetry into a predicted time to maintenance
(TTM) per component, turns those predictions into a parts list, and turns the
parts list into stock moves a planner can approve - with the model's
uncertainty carried through every step.

## Run it

```powershell
.\run_demo.ps1 -Rebuild    # build the pipeline, then start API + web
.\run_demo.ps1             # just start (uses the existing DuckDB)
```

Then open <http://localhost:5173/>.

Rebuilding runs: synthetic fleet -> features -> Weibull prior -> AFT survival
model -> interval calibration -> SHAP -> assumption cards -> repair kits ->
Monte Carlo parts demand -> inventory policy -> transfer-vs-buy optimiser ->
DuckDB. Individual stages:

```powershell
python -m pipeline.build_all --from planning   # skip the model refit
python -m pipeline.build_all --only load       # just reload DuckDB
```

## The two checks that matter

```powershell
# 1. Does uncertainty actually reach the stock decision? (the core claim)
python -m planning.inventory_policy --sensitivity
#    class-A safety stock must RISE when the TTM band widens.

# 2. Is the model honest per component type?
python -m models.survival_xgb --evaluate
#    prints a per-type table and flags any type below 0.3 rank correlation.
#    The fleet-level metric twice hid total failure on the components that
#    matter most, so this table is not optional.
```

## The physics-informed network (PINN)

`models/pinn.py`. The governing equations of the refrigeration cycle replace
labelled data: nothing in the field carries a "the condenser is 23% fouled"
label, so a supervised model cannot be trained to emit one, but the physics
can supply that supervision instead.

```powershell
python -m models.pinn --identifiability   # which parameters the sensors can resolve
python -m models.pinn --gradcheck         # analytic vs finite-difference gradients
python -m models.pinn --sample            # ~9 min: sample the twin
python -m models.pinn --train-surrogate   # stage 1
python -m models.pinn --train-inverse     # stage 2, no labels
python -m models.pinn --evaluate          # inferred state vs known damage
```

Two stages. **S** maps (fault state, conditions) to signals, trained only on
output from the twin - it exists because the twin itself is not
differentiable, solving nested brentq over CoolProp lookups. **F** inverts it:
given the sensors, infer the fault state, with the loss being the physics
residual `|| S(F(obs)) - obs ||²` and no labels anywhere.

It is numpy with analytic gradients, checked against finite differences to
1e-6. A deep-learning framework would have cost ~300 MB resident for a few
hundred parameters, in the same process as the survival booster, on a box
with under 600 MB free.

**Three parameters, not five, and that was measured.** `wear_frac` moves these
four signals by under one noise sigma over its whole range and accounts for
0.996 of the smallest singular direction - compressor wear shows up in power
draw, and this fleet's current channel is uncalibrated. `foul_frac` and
`airflow_frac` are aliased through discharge pressure. The remaining three
have a Jacobian condition number of 12.6-14.3 across the operating envelope.

**Result: one of three parameters works.**

| inferred | vs its own true damage | vs ambient | pinned at bound |
|---|---|---|---|
| **Refrigerant charge** | **r = 0.79** | 0.31 | 0.1% |
| Condenser degradation | r = 0.23 | **-0.78** | 60% |
| Evaporator degradation | r = 0.11 | -0.62 | 70% |

Refrigerant charge is a genuine virtual sensor - r = 0.79 against the
generator's true refrigerant-circuit damage, clean against the other two
modes, and barely ever saturated. It works because superheat carries charge,
and superheat is a channel the twin reproduces to 0.08 of the sensor noise.

The other two fail for the reason recorded in the next section: they read the
weather and saturate against their bounds. The held-out physics residual is
~106 squared noise sigmas where a sound forward model would give 1.0, so no
reachable fault state explains this fleet's pressures. The UI shows all three
and labels which survived validation, because hiding the two failures would
be the dishonest option and dropping them would hide that the method was
tried.

One debugging note worth keeping: the surrogate sat at a 4-sigma error floor
until `asset_type` was added as an input. Each machine family has its own
design point, so the same fault state and weather give different pressures on
a chiller and an RTU, and the network could not tell them apart - 7.3 sigmas
on chillers against 2.5 on split units. With the family supplied it fits to
**0.09-0.17 sigmas**, about 25x finer than the sensors. `cap_ratio` went the
other way and was removed: the twin's equations are homogeneous in capacity,
so a 40 kW and a 560 kW chiller return bit-identical pressures.

## One thing we tried that did not work

The asset twin (`sim/asset_twin.py`) was built to supply a residual feature:
`observed - healthy(conditions)` should isolate degradation from the weather,
and so correlate more with damage and less with ambient than the raw channel.

It does not, on this data, and the check is in the repo rather than the claim:

```powershell
python -m sim.twin_residual_check --n 1200          # and --asset-type rtu, vrf, ...
```

| signal | r(ambient) raw -> residual | r(damage) raw -> residual |
|---|---|---|
| discharge_p | +0.72 -> **-0.87** | +0.45 -> **+0.22** |
| suction_p | +0.09 -> **+0.53** | -0.44 -> **-0.35** |
| superheat | -0.00 -> -0.00 | +0.42 -> +0.42 |
| subcool | +0.04 -> +0.04 | +0.45 -> +0.45 |

0 of 4 signals improve, on all four machine families. The reason is a
property of the data, not of the method: the synthetic fleet's head pressure
moves 26 kPa per K of ambient, while R32 saturation physics - which is what
the twin computes - moves about 60 kPa/K. Subtracting a 60 kPa/K model from a
26 kPa/K signal over-corrects, so the residual carries *more* weather than
the raw channel, with the sign flipped. Superheat and subcool sit at fixed
setpoints in the twin, so their residual is the raw signal minus a constant,
and a tree ensemble is invariant to that.

So the twin is **not** a model feature here. It earns its place driving the
what-if panel and the asset screen's healthy baseline, where a physically
correct answer to "what would a sound machine read in this weather" is
exactly what is wanted. Against real telemetry, where head pressure does
follow saturation, the residual would remove a real confounder - but that is
a claim about other data, so it is not made as a result of this one.

The honest consequence: the three planted hero compressors are still
mispredicted, and the twin residual was the intended fix for them.

## Layout

| Path | What it is |
|---|---|
| `contracts/` | Pydantic API contract + the derivation rules shared by pipeline and API |
| `sim/` | Seeded synthetic fleet, the world catalogue, the Azure PdM loader |
| `features/` | Causal rolling features + the AFT snapshot sampler |
| `models/` | Weibull prior, XGBoost AFT, interval calibration, SHAP, assumption cards |
| `planning/` | Repair kits, Monte Carlo demand, inventory policy, optimiser |
| `pipeline/` | `build_all.py` - one command, whole chain, into DuckDB |
| `api/` | FastAPI over DuckDB (thin SELECTs) + telemetry replay socket |
| `web/` | React + Vite + Tailwind; four screens |

## Notes on the data

The synthetic fleet exists because no public dataset links HVAC telemetry,
service history, BOM, hub stock and supplier lead times together, and A01 needs
all five. Degradation is damage accumulation against a Weibull threshold, driven
by thermal, cycling and load stress - validated: hot climate zones fail 45-65%
more often than mild ones, and the fault signatures match the textbook
directions (fouling raises head pressure and drops COP; a leak drops suction and
raises superheat).

The **Microsoft/Azure Predictive Maintenance** set (100 machines, 876k telemetry
rows, 761 failures, 77% censoring) validates the same survival pipeline on real
component-level failure history. It is not HVAC, and the pitch says so - its job
is to show the method holds on real censored data.

## What's committed vs regenerated

The repo carries the **results**, not the bulk, so you can inspect real numbers
without running anything:

```python
import pandas as pd
pd.read_parquet("data/synthetic/recommendations.parquet")   # the 121 cards
pd.read_parquet("data/synthetic/component_ttm.parquet")     # P10/P50/P90 per component
pd.read_parquet("data/synthetic/stock_positions.parquet")   # reorder points, excess
pd.read_parquet("models/artifacts/eval_by_type.parquet")    # per-type honesty table
```

`models/artifacts/` also holds the **trained model** (`survival_xgb.json`) plus
its calibration, so the reported scores are verifiable without a refit.

Excluded, and why:

| Not committed | Size | Reason |
|---|---|---|
| `data/servicesense.duckdb` | 232 MB | Over GitHub's 100 MB per-file hard limit; fully derived |
| `data/synthetic/telemetry_daily.parquet` | 182 MB | Same |
| `train_snapshots`, `asset_features` | 72 MB | Regenerable intermediates |
| `data/raw/azure_pdm/` | 76 MB | Third-party public data; `sim/azure_pdm.py` re-downloads it |

All of it rebuilds deterministically from the frozen seed:

```powershell
.\run_demo.ps1 -Rebuild     # ~6 min, then the UI is live
```

### Or skip the rebuild

The prebuilt database is published as a **GitHub Release asset** — Release
assets allow up to 2 GB each, unlike the 100 MB per-file limit on committed
files:

1. Download `servicesense.duckdb` from the latest release into `data/`
2. `.\run_demo.ps1` — no `-Rebuild` flag
3. Open <http://localhost:5173/>

To publish or refresh that release:

```powershell
.\publish_release.ps1          # the database only (~232 MB)
.\publish_release.ps1 -Full    # plus the heavy parquets, to re-run features
```

It is idempotent — re-running replaces the assets on an existing tag rather
than failing — and refuses early with a clear message if `gh` is not
authenticated or the artefacts have not been built yet.
