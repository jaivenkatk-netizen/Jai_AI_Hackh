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
