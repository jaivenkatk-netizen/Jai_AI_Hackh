"""One command, whole chain, into DuckDB.

    python -m pipeline.build_all

Runs data -> features -> models -> planning in order and loads every artefact
into a single DuckDB file that the API reads with thin SELECTs. Precomputing
like this is deliberate: it keeps the demo fast and crash-proof, and it means a
model bug is a re-run rather than a debugging session inside a web request. The
only live compute path in the whole system is the single-asset what-if.

Stages are skippable so an iteration on (say) the optimiser does not re-run a
20-minute model fit:

    python -m pipeline.build_all --from planning
    python -m pipeline.build_all --only load
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
DB_PATH = ROOT / "data" / "servicesense.duckdb"

# Ordered pipeline. Each stage is a module run with -m so it keeps its own CLI.
STAGES: list[tuple[str, list[str]]] = [
    ("simulate", ["sim.simulate_fleet"]),
    ("features", ["features.build_features"]),
    ("weibull", ["models.weibull_prior"]),
    ("survival", ["models.survival_xgb"]),
    ("conformal", ["models.conformal"]),
    ("explain", ["models.explain"]),
    ("assumptions", ["models.assumptions"]),
    ("kits", ["planning.repair_kits"]),
    ("demand", ["planning.parts_demand"]),
    ("policy", ["planning.inventory_policy"]),
    ("planning", ["planning.optimizer"]),
]
STAGE_NAMES = [s for s, _ in STAGES] + ["load"]

# Parquet artefacts to register as DuckDB tables. Missing files are skipped with
# a warning rather than failing the load - the Azure PdM tables, for instance,
# are absent when the machine is offline, and synthetic-only is a valid mode.
TABLES = [
    "hubs", "sites", "assets", "components", "components_truth",
    "telemetry_daily", "telemetry_hourly", "work_orders",
    "parts", "suppliers", "bom", "inventory", "hub_transfers",
    "component_predictions", "component_ttm", "component_drivers",
    "repair_kits", "failure_mode_mix", "van_kits",
    "parts_demand", "parts_demand_contrib", "stock_positions", "recommendations",
    "asset_features", "score_snapshots",
    "pdm_machines", "pdm_lives", "pdm_daily", "pdm_train",
]

# Views the API leans on, defined once here so the endpoints stay trivial.
VIEWS: dict[str, str] = {
    # Fleet health: the worst component per asset, with its band and badge.
    "v_asset_risk": """
        SELECT
            a.asset_id, a.asset_label, a.asset_type, a.model, a.site_name AS site,
            a.hub_id, a.region, a.criticality, a.capacity_kw, a.climate_zone,
            t.component_id AS top_risk_component_id,
            t.component_type AS top_risk_component_type,
            t.p10_days, t.p50_days, t.p90_days, t.beyond_horizon,
            t.p_fail_14d, t.p_fail_90d, t.confidence, t.confidence_reasons,
            t.interval_width_days, t.data_quality,
            CASE
                WHEN t.p_fail_14d >= CASE a.criticality WHEN 'A' THEN 0.30 ELSE 0.50 END
                     OR t.p50_days <= 14 THEN 'urgent'
                WHEN t.p50_days <= 90 THEN 'planned'
                ELSE 'forecast'
            END AS bucket
        FROM assets a
        JOIN (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY asset_id ORDER BY p50_days ASC) AS rn
            FROM component_ttm
        ) t ON t.asset_id = a.asset_id AND t.rn = 1
    """,
    # Every component with its bucket - what the parts drill-down joins against.
    "v_component_risk": """
        SELECT
            t.*, a.hub_id, a.region, a.site_name AS site, a.criticality AS asset_criticality,
            a.asset_label, a.asset_type, a.size_class, a.refrigerant,
            CASE
                WHEN t.p_fail_14d >= CASE a.criticality WHEN 'A' THEN 0.30 ELSE 0.50 END
                     OR t.p50_days <= 14 THEN 'urgent'
                WHEN t.p50_days <= 90 THEN 'planned'
                ELSE 'forecast'
            END AS bucket
        FROM component_ttm t
        JOIN assets a ON a.asset_id = t.asset_id
    """,
    # Hub rollup for the Command Center map.
    "v_hub_risk": """
        SELECT
            h.hub_id, h.name, h.region, h.lat, h.lon,
            COUNT(DISTINCT CASE WHEN r.bucket <> 'forecast' THEN r.asset_id END) AS assets_at_risk,
            COUNT(DISTINCT CASE WHEN r.bucket = 'urgent' THEN r.asset_id END) AS urgent_jobs,
            COALESCE(s.projected_stockouts, 0) AS projected_stockouts,
            COALESCE(s.excess_value, 0) AS excess_value
        FROM hubs h
        LEFT JOIN v_asset_risk r ON r.hub_id = h.hub_id
        LEFT JOIN (
            SELECT hub_id,
                   SUM(CASE WHEN shortfall_vs_p90 > 0 THEN 1 ELSE 0 END) AS projected_stockouts,
                   SUM(excess_value) AS excess_value
            FROM stock_positions GROUP BY hub_id
        ) s ON s.hub_id = h.hub_id
        GROUP BY h.hub_id, h.name, h.region, h.lat, h.lon, s.projected_stockouts, s.excess_value
    """,
    # Urgent job queue, flagged where a needed part is short at the hub.
    "v_urgent_jobs": """
        SELECT
            c.component_id, c.asset_id, c.asset_label, c.site, c.hub_id,
            c.component_type, c.p10_days, c.p50_days, c.p90_days,
            c.p_fail_14d, c.confidence, c.confidence_reasons, c.assumptions,
            EXISTS (
                SELECT 1 FROM parts_demand_contrib pc
                JOIN stock_positions sp
                  ON sp.part_no = pc.part_no AND sp.hub_id = pc.hub_id
                WHERE pc.component_id = c.component_id AND sp.shortfall_vs_p90 > 0
            ) AS parts_short
        FROM v_component_risk c
        WHERE c.bucket = 'urgent'
    """,
}


def run_stage(name: str, module: str, extra: list[str] | None = None) -> float:
    t0 = time.perf_counter()
    print(f"\n{'=' * 72}\n[{name}] python -m {module}\n{'=' * 72}", flush=True)
    cmd = [sys.executable, "-u", "-m", module, *(extra or [])]
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"stage '{name}' failed with exit code {proc.returncode}")
    dt = time.perf_counter() - t0
    print(f"[{name}] ok in {dt:.1f}s", flush=True)
    return dt


def load_duckdb(data: Path, db_path: Path) -> dict[str, int]:
    """Register every Parquet artefact as a table, then build the API views."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    counts: dict[str, int] = {}
    try:
        for name in TABLES:
            pq = data / f"{name}.parquet"
            if not pq.exists():
                print(f"  - {name:26s} (absent, skipped)")
                continue
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet(?)", [str(pq)])
            n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            counts[name] = int(n)
            print(f"  + {name:26s} {n:>10,} rows")

        for vname, sql in VIEWS.items():
            con.execute(f"CREATE OR REPLACE VIEW {vname} AS {sql}")
            n = con.execute(f"SELECT COUNT(*) FROM {vname}").fetchone()[0]
            counts[vname] = int(n)
            print(f"  v {vname:26s} {n:>10,} rows")
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return counts


def gate_a_report(db_path: Path) -> None:
    """The Gate A check: can a telemetry reading reach an approvable action?"""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        print(f"\n{'=' * 72}\nGATE A - telemetry to approvable action\n{'=' * 72}")
        rows = con.execute(
            """
            SELECT rec_id, action, part_no, qty, from_hub_id, to_hub_id,
                   ROUND(need_in_days, 1) AS need_in_days, confidence
            FROM recommendations ORDER BY need_in_days LIMIT 5
            """
        ).df()
        print(rows.to_string(index=False))

        card = con.execute(
            """
            SELECT rec_id, action, why, evidence, assumptions, confidence
            FROM recommendations
            WHERE action = 'transfer'
            ORDER BY need_in_days LIMIT 1
            """
        ).df()
        if len(card):
            c = card.iloc[0]
            print(f"\nExample card [{c.rec_id}] {c.action.upper()} - confidence {c.confidence}")
            print(f"  WHY: {c.why}")
            import json as _json

            for e in _json.loads(c.evidence):
                print(f"  - {e}")
            print("  ASSUMPTIONS:")
            for a in _json.loads(c.assumptions):
                print(f"  - {a['label']}: {a['text']}  ({a.get('basis')})")

        summary = con.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM assets) AS assets,
                (SELECT COUNT(*) FROM component_ttm) AS components_scored,
                (SELECT COUNT(*) FROM v_urgent_jobs) AS urgent_jobs,
                (SELECT COUNT(*) FROM parts_demand) AS demand_lines,
                (SELECT COUNT(*) FROM recommendations) AS recommendations,
                (SELECT COUNT(*) FROM recommendations WHERE action='transfer') AS transfers,
                (SELECT COUNT(*) FROM recommendations WHERE avoided_action='expedite') AS expedites_avoided,
                (SELECT ROUND(SUM(excess_value)) FROM stock_positions) AS excess_value
            """
        ).df()
        print("\n" + summary.to_string(index=False))

        ok = (
            int(summary["recommendations"].iloc[0]) > 0
            and int(summary["urgent_jobs"].iloc[0]) > 0
            and int(summary["components_scored"].iloc[0]) > 0
        )
        print(f"\nGATE A: {'PASS' if ok else 'FAIL'}")
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the whole ServiceSense pipeline")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--from", dest="from_stage", choices=STAGE_NAMES, default=None,
                    help="start at this stage")
    ap.add_argument("--only", choices=STAGE_NAMES, default=None, help="run a single stage")
    ap.add_argument("--skip-sim", action="store_true",
                    help="reuse the existing synthetic fleet (equivalent to --from features)")
    args = ap.parse_args()

    stages = STAGES
    if args.only:
        stages = [s for s in STAGES if s[0] == args.only]
    else:
        start = args.from_stage or ("features" if args.skip_sim else None)
        if start and start != "load":
            idx = [s for s, _ in STAGES].index(start)
            stages = STAGES[idx:]
        elif start == "load":
            stages = []

    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    for name, (module, *rest) in [(n, m) for n, m in stages]:
        timings[name] = run_stage(name, module, rest)

    if args.only in (None, "load"):
        print(f"\n{'=' * 72}\n[load] -> {args.db}\n{'=' * 72}")
        load_duckdb(args.data, args.db)
        gate_a_report(args.db)

    total = time.perf_counter() - t0
    if timings:
        print("\nstage timings:")
        for k, v in timings.items():
            print(f"  {k:14s} {v:7.1f}s")
    print(f"\ntotal {total:.1f}s")


if __name__ == "__main__":
    main()
