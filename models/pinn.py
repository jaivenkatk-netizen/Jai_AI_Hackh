"""Physics-informed inverse estimation of an asset's fault state.

What this is, stated precisely, because "PINN" covers a lot of ground: the
governing equations of the vapour-compression cycle replace labelled data as
the training signal. No row of the fleet's telemetry carries a "the condenser
is 23% fouled" label - nothing does, in the field either - so a supervised
model cannot be trained to output one. The physics can supply that
supervision instead.

The chain:

    Stage 1   S : (fault state, conditions) -> (pressures, superheat, subcool)
              A small network trained ONLY on output from sim/asset_twin.py.
              It exists because the twin itself is not differentiable: it
              solves nested brentq over CoolProp property lookups, so there
              is no gradient to backpropagate. S is a differentiable stand-in
              for the physics.

    Stage 2   F : (observed signals, conditions) -> fault state
              Trained on real fleet telemetry with NO labels. The loss is the
              physics residual,

                  L = || S(F(obs, cond), cond) - obs ||^2 / sigma^2
                      + lambda * || F(obs, cond) - healthy ||^2

              The first term says: whatever fault state you infer, running the
              physics forward from it must reproduce what the sensors actually
              read. The second is a weak prior toward a healthy machine, so a
              signal that carries no information returns "healthy" rather than
              something arbitrary.

Why bother, when `residuals()` already compares observed against healthy?
Because subtraction throws away the constraint. The residual is four loose
numbers; the inverse solve is forced onto the 3-dimensional manifold of
physically reachable fault states, pools all four signals at once, and
returns something dimensionless and interpretable - a fouling fraction, a
charge fraction - instead of a pressure offset whose scale depends on the
machine. `sim/twin_residual_check.py` records why the subtraction approach
failed on this fleet; this is the constrained alternative, and it is measured
the same way in `--evaluate` rather than asserted.

Why three parameters and not five. Measured, not assumed (the Jacobian study
is reproduced by `--identifiability`):

  * wear_frac is invisible. Over its full plausible range it moves these four
    signals by under one noise sigma, and it accounts for 0.996 of the
    smallest singular direction. Compressor wear shows up in power draw, and
    this fleet's current channel is on an uncalibrated scale, so wear is left
    out rather than fitted to noise.
  * foul_frac and airflow_frac are aliased - both act on the condenser
    conductance and move discharge pressure in opposite directions with
    little else to tell them apart. They are folded into one `cond_deg`.

That leaves cond_deg, charge_frac and evap_deg, for which the Jacobian has a
condition number of 12.6 to 14.3 across the operating envelope and every
parameter clears the noise floor.

Everything here is numpy with analytic gradients, checked against finite
differences by `--gradcheck`. The networks have a few hundred parameters
each; importing a deep-learning framework would have cost ~300 MB of
resident memory on a box with under 600 MB free, in the same process that
already holds the survival booster.

Run:

    python -m models.pinn --identifiability     # the Jacobian study above
    python -m models.pinn --sample              # ~7 min, writes twin samples
    python -m models.pinn --train-surrogate     # stage 1, reports held-out error
    python -m models.pinn --gradcheck           # analytic vs finite-difference
    python -m models.pinn --train-inverse       # stage 2, physics loss, no labels
    python -m models.pinn --evaluate            # inferred state vs true damage
    python -m models.pinn --infer               # write fault_state table
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from sim.asset_twin import solve, spec_for_asset

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "models" / "artifacts"
DATA = ROOT / "data" / "synthetic"

# The four signals the twin reproduces on an absolute scale. comp_current and
# cop are excluded for the reasons given in sim/asset_twin.py.
SIGNALS = ("suction_p", "discharge_p", "superheat", "subcool")

# Measurement noise per signal, from the generator's own SIGNAL_NOISE. The
# physics residual is weighted by these so a 0.45 K superheat error and a
# 22 kPa discharge error count the same.
SIGMA = np.array([9.0, 22.0, 0.45, 0.35])

# The inferred fault state, with the range each parameter is allowed to take.
FAULTS = ("cond_deg", "charge_frac", "evap_deg")
FAULT_LO = np.array([0.0, 0.65, 0.0])
FAULT_HI = np.array([0.60, 1.05, 0.40])
FAULT_HEALTHY = np.array([0.0, 1.0, 0.0])

# Which inferred parameters survived validation against the generator's known
# damage, from `--evaluate` on 40,000 held-out asset-days. Carried alongside
# the numbers so nothing unvalidated reaches a screen as if it were a reading.
#
#   charge_frac   r = 0.786 against true refrigerant-circuit damage, clean
#                 against the other two modes (-0.08, 0.27), only 0.1% of
#                 assets pinned at a bound. A real virtual sensor.
#   cond_deg      r = 0.229, but r = -0.78 against ambient and 60% of assets
#                 pinned at the upper bound - it is reading the weather and
#                 saturating, not measuring fouling.
#   evap_deg      r = 0.111, 70% pinned at the upper bound. Not recovered.
#
# The two failures share the root cause recorded in sim/twin_residual_check.py:
# the fleet's condition response is not thermodynamically consistent, so the
# pressure channels cannot be inverted. Superheat is the exception, and
# superheat is what carries charge - which is why that one parameter works.
FAULT_VALIDATED: dict[str, bool] = {"cond_deg": False, "charge_frac": True, "evap_deg": False}
FAULT_LABEL = {
    "cond_deg": "Condenser degradation",
    "charge_frac": "Refrigerant charge",
    "evap_deg": "Evaporator degradation",
}

# Operating conditions, which are known rather than inferred.
#
# Two corrections here, both measured rather than guessed:
#
#   * `cap_ratio` was dropped. The twin's equations are homogeneous in rated
#     capacity - displacement and both conductances scale together - so the
#     saturation temperatures and pressures are *exactly* capacity-invariant.
#     Solving a 40 kW and a 560 kW chiller at the same conditions returns
#     bit-identical pressures. Only the absolute duty and power scale. It was
#     a pure nuisance dimension, costing sample density for nothing.
#   * The asset family was added. `spec_for_asset` gives each family its own
#     design point, so the same fault state and weather produce quite
#     different pressures on a chiller (7 C leaving water) and an RTU (13 C
#     leaving air). Without this input the surrogate could not tell the
#     families apart and carried an irreducible error of 7.3 noise sigmas on
#     chillers against 2.5 on split units. That was the entire error floor.
ASSET_TYPES = ("chiller", "rtu", "vrf", "split_heat_pump")
CONDS = ("ambient_c", "load_frac", "is_r410a") + tuple(f"is_{t}" for t in ASSET_TYPES)


def add_cond_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the asset-family indicator columns the surrogate needs."""
    out = df.copy()
    at = out["asset_type"].astype(str)
    for t in ASSET_TYPES:
        out[f"is_{t}"] = (at == t).astype(float)
    # Anything outside the four modelled families (an air-only AHU) falls back
    # to the chiller design point, which is what spec_for_asset does too.
    unknown = ~at.isin(ASSET_TYPES)
    if unknown.any():
        out.loc[unknown, "is_chiller"] = 1.0
    return out


def fault_to_spec_kwargs(cond_deg: float, charge_frac: float, evap_deg: float) -> dict:
    """Map the reduced fault state onto the twin's parameters.

    `cond_deg` drives fouling and fan airflow together, in the fixed ratio
    that the identifiability study showed they cannot be separated at.
    """
    return {
        "foul_frac": float(cond_deg),
        "airflow_frac": float(1.0 - 0.55 * cond_deg),
        "charge_frac": float(charge_frac),
        "evap_foul_frac": float(evap_deg),
        "wear_frac": 0.0,
    }


# --------------------------------------------------------------------------
# A small MLP, numpy, analytic gradients
# --------------------------------------------------------------------------


class MLP:
    """Fully-connected net with tanh hidden layers.

    `out_lo`/`out_hi` turn the output layer into a scaled sigmoid, which is
    how the inverse network is kept inside the physically reachable range.
    A clamp after the fact would have killed the gradient at the bounds.
    """

    def __init__(
        self,
        sizes: list[int],
        seed: int = 0,
        out_lo: np.ndarray | None = None,
        out_hi: np.ndarray | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(sizes) - 1):
            # Xavier: keeps tanh in its responsive band at initialisation
            scale = np.sqrt(2.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.normal(0, scale, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        self.out_lo = out_lo
        self.out_hi = out_hi
        self._m = [np.zeros_like(p) for p in self.params()]
        self._v = [np.zeros_like(p) for p in self.params()]
        self._t = 0

    def params(self) -> list[np.ndarray]:
        return self.W + self.b

    # -- forward ---------------------------------------------------------
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        acts = [x]
        h = x
        for i in range(len(self.W) - 1):
            h = np.tanh(h @ self.W[i] + self.b[i])
            acts.append(h)
        z = h @ self.W[-1] + self.b[-1]
        cache = {"acts": acts, "z_out": z}
        if self.out_lo is None:
            return z, cache
        s = 1.0 / (1.0 + np.exp(-z))
        cache["sig"] = s
        return self.out_lo + (self.out_hi - self.out_lo) * s, cache

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    # -- backward --------------------------------------------------------
    def backward(
        self, cache: dict, dy: np.ndarray, want_dx: bool = False
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Gradients of the loss wrt parameters, and optionally wrt the input.

        `want_dx` is what makes stage 2 work: the physics residual's gradient
        has to pass through the frozen surrogate and out of its input layer to
        reach the inverse network.
        """
        acts = cache["acts"]
        if self.out_lo is not None:
            s = cache["sig"]
            dy = dy * (self.out_hi - self.out_lo) * s * (1.0 - s)

        dW = [np.zeros_like(w) for w in self.W]
        db = [np.zeros_like(v) for v in self.b]
        n = len(self.W)

        d = dy
        for i in range(n - 1, -1, -1):
            a_in = acts[i]
            dW[i] = a_in.T @ d
            db[i] = d.sum(axis=0)
            if i == 0:
                dx = d @ self.W[0].T if want_dx else None
                break
            d = (d @ self.W[i].T) * (1.0 - acts[i] ** 2)
        else:  # pragma: no cover - n >= 1 always
            dx = None
        return dW + db, dx

    # -- Adam ------------------------------------------------------------
    def step(self, grads: list[np.ndarray], lr: float, wd: float = 0.0) -> None:
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        ps = self.params()
        for k, (p, g) in enumerate(zip(ps, grads)):
            if wd and k < len(self.W):  # decay weights, never biases
                g = g + wd * p
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            mh = self._m[k] / (1 - b1**self._t)
            vh = self._v[k] / (1 - b2**self._t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        d = {
            "sizes": self.sizes,
            "W": [w.tolist() for w in self.W],
            "b": [v.tolist() for v in self.b],
        }
        if self.out_lo is not None:
            d["out_lo"] = self.out_lo.tolist()
            d["out_hi"] = self.out_hi.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> MLP:
        lo = np.array(d["out_lo"]) if "out_lo" in d else None
        hi = np.array(d["out_hi"]) if "out_hi" in d else None
        m = cls(d["sizes"], out_lo=lo, out_hi=hi)
        m.W = [np.array(w) for w in d["W"]]
        m.b = [np.array(v) for v in d["b"]]
        return m


class Scaler:
    """Standardise to zero mean and unit variance, and remember how."""

    def __init__(self, x: np.ndarray | None = None) -> None:
        if x is None:
            self.mu = self.sd = None
            return
        self.mu = x.mean(axis=0)
        self.sd = x.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.sd + self.mu

    def to_dict(self) -> dict:
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> Scaler:
        s = cls()
        s.mu = np.array(d["mu"])
        s.sd = np.array(d["sd"])
        return s


# --------------------------------------------------------------------------
# Stage 0: identifiability
# --------------------------------------------------------------------------


def identifiability() -> None:
    """Which fault parameters can these four signals actually resolve?"""
    sp = spec_for_asset("chiller", 280.0, "R32")
    full = ("foul_frac", "wear_frac", "charge_frac", "airflow_frac", "evap_foul_frac")
    base = dict(foul_frac=0.15, wear_frac=0.10, charge_frac=0.92, airflow_frac=0.90,
                evap_foul_frac=0.10)
    spans = dict(foul_frac=0.5, wear_frac=0.3, charge_frac=0.3, airflow_frac=0.4,
                 evap_foul_frac=0.3)

    def y_of(d: dict, amb: float = 32.0, lf: float = 0.6) -> np.ndarray:
        s = solve(replace(sp, **d), amb, lf)
        return np.array([s.suction_p_kpa, s.discharge_p_kpa, s.superheat_k, s.subcool_k])

    y0 = y_of(base)
    J = np.zeros((4, len(full)))
    for j, f in enumerate(full):
        d = dict(base)
        h = 0.02 * spans[f]
        d[f] = base[f] + h
        J[:, j] = (y_of(d) - y0) / (h / spans[f])
    Jn = J / SIGMA[:, None]

    print("\nFull 5-parameter problem. Signal change over each parameter's full")
    print("range, in units of measurement noise:\n")
    print(f"{'':<14}" + "".join(f"{f[:12]:>13}" for f in full))
    for i, s in enumerate(SIGNALS):
        print(f"{s:<14}" + "".join(f"{v:>13.1f}" for v in Jn[i]))
    sv, vt = np.linalg.svd(Jn)[1], np.linalg.svd(Jn)[2]
    print(f"\nsingular values: {np.array2string(sv, precision=1)}")
    print(f"condition number: {sv[0] / max(sv[-1], 1e-12):.1f}")
    print("\nsmallest singular direction - the combination the data cannot see:")
    for f, w in sorted(zip(full, vt[-1]), key=lambda t: -abs(t[1])):
        print(f"   {f:<16} {w:+.3f}")

    print("\n\nReduced 3-parameter problem actually used:\n")
    print(f"{'ambient':<9} {'load':<6} " + "".join(f"{n:>13}" for n in FAULTS) + f" {'cond#':>7}")
    for amb, lf in [(22.0, 0.4), (32.0, 0.6), (42.0, 0.9)]:
        p = np.array([0.15, 0.92, 0.10])
        y0 = y_of(fault_to_spec_kwargs(*p), amb, lf)
        Jr = np.zeros((4, 3))
        spans_r = FAULT_HI - FAULT_LO
        for j in range(3):
            q = p.copy()
            h = 0.02 * spans_r[j]
            q[j] += h
            Jr[:, j] = (y_of(fault_to_spec_kwargs(*q), amb, lf) - y0) / (h / spans_r[j])
        Jrn = Jr / SIGMA[:, None]
        svr = np.linalg.svd(Jrn, compute_uv=False)
        cols = "".join(f"{np.linalg.norm(Jrn[:, j]):>13.1f}" for j in range(3))
        print(f"{amb:<9.0f} {lf:<6.1f} {cols} {svr[0] / svr[-1]:>7.1f}")
    print("\nColumn values are total signal movement in noise sigmas; a parameter")
    print("below about 1.0 cannot be estimated from these signals at all.")


# --------------------------------------------------------------------------
# Stage 1: sample the twin, fit the differentiable surrogate
# --------------------------------------------------------------------------


def sample_twin(n: int = 6000, seed: int = 17) -> pd.DataFrame:
    """Latin-hypercube sample of the twin over fault state x conditions."""
    rng = np.random.default_rng(seed)

    def lhs(k: int) -> np.ndarray:
        cut = (np.arange(n) + rng.random(n)) / n
        out = np.empty((n, k))
        for j in range(k):
            out[:, j] = rng.permutation(cut)
        return out

    u = lhs(5)
    cond_deg = FAULT_LO[0] + u[:, 0] * (FAULT_HI[0] - FAULT_LO[0])
    charge = FAULT_LO[1] + u[:, 1] * (FAULT_HI[1] - FAULT_LO[1])
    evap_deg = FAULT_LO[2] + u[:, 2] * (FAULT_HI[2] - FAULT_LO[2])
    ambient = 10.0 + u[:, 3] * 42.0  # the fleet spans 11.5 to 52.7 C
    load = 0.05 + u[:, 4] * 0.95

    types = rng.choice(ASSET_TYPES, size=n)
    refrig = rng.choice(["R32", "R410A"], size=n)

    rows, failed, t0 = [], 0, time.perf_counter()
    for i in range(n):
        # Capacity is fixed: the pressures do not depend on it (see CONDS).
        spec = spec_for_asset(str(types[i]), 280.0, str(refrig[i]))
        kw = fault_to_spec_kwargs(cond_deg[i], charge[i], evap_deg[i])
        try:
            s = solve(replace(spec, **kw), float(ambient[i]), float(load[i]))
        except Exception:
            failed += 1
            continue
        if not s.converged:
            failed += 1
            continue
        rows.append(
            {
                "cond_deg": cond_deg[i],
                "charge_frac": charge[i],
                "evap_deg": evap_deg[i],
                "ambient_c": ambient[i],
                "load_frac": load[i],
                "is_r410a": 1.0 if refrig[i] == "R410A" else 0.0,
                "asset_type": types[i],
                "suction_p": s.suction_p_kpa,
                "discharge_p": s.discharge_p_kpa,
                "superheat": s.superheat_k,
                "subcool": s.subcool_k,
            }
        )
        if (i + 1) % 500 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{n} solved, {failed} dropped, {el:.0f}s elapsed", flush=True)

    df = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "twin_samples.parquet"
    df.to_parquet(out, index=False)
    print(f"\n{len(df)} samples ({failed} non-convergent, dropped) -> {out}")
    return df


def train_surrogate(epochs: int = 1200, hidden: int = 96, seed: int = 3) -> None:
    """Fit S: (fault state, conditions) -> signals, on twin output only."""
    df = add_cond_columns(pd.read_parquet(DATA / "twin_samples.parquet"))
    x = df[list(FAULTS) + list(CONDS)].to_numpy(dtype=np.float64)
    y = df[list(SIGNALS)].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_val = max(int(0.15 * len(x)), 1)
    va, tr = idx[:n_val], idx[n_val:]

    sx, sy = Scaler(x[tr]), Scaler(y[tr])
    xt, yt = sx(x[tr]), sy(y[tr])
    xv, yv = sx(x[va]), sy(y[va])

    net = MLP([xt.shape[1], hidden, hidden, hidden, yt.shape[1]], seed=seed)

    # Train in the units the result is judged in. Plain MSE on standardised
    # targets weights the four signals equally, but one standardised unit of
    # discharge pressure is 27 noise sigmas where one of superheat is 11, so
    # equal weighting quietly optimised the wrong thing and the pressures
    # plateaued at 4 sigmas. Weighting by (sd/sigma)^2 makes the objective
    # mean squared error in noise sigmas, which is what gets reported.
    w_out = (sy.sd / SIGMA) ** 2
    w_out = w_out / w_out.mean()

    batch = 256
    best, best_state = np.inf, None
    for ep in range(epochs):
        order = rng.permutation(len(xt))
        lr = 3e-3 * (0.5 ** (ep / 220))
        for k in range(0, len(order), batch):
            b = order[k : k + batch]
            pred, cache = net.forward(xt[b])
            d = 2.0 * (pred - yt[b]) * w_out / len(b)
            grads, _ = net.backward(cache, d)
            net.step(grads, lr, wd=1e-6)
        if ep % 50 == 0 or ep == epochs - 1:
            vp = net(xv)
            # Report in noise sigmas: the number that says whether the
            # surrogate is accurate enough to invert through.
            err = np.abs(sy.inverse(vp) - sy.inverse(yv)) / SIGMA
            mse = float((((vp - yv) ** 2) * w_out).mean())
            if mse < best:
                best, best_state = mse, (net.to_dict(), ep)
            print(
                f"  epoch {ep:4d}  val mse {mse:.5f}  "
                f"mean |err| in sigmas: "
                + " ".join(f"{s}={e:.2f}" for s, e in zip(SIGNALS, err.mean(axis=0))),
                flush=True,
            )

    net = MLP.from_dict(best_state[0])
    vp = sy.inverse(net(xv))
    truth = sy.inverse(yv)
    err = np.abs(vp - truth) / SIGMA

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "pinn_surrogate.json").write_text(
        json.dumps(
            {
                "net": net.to_dict(),
                "scaler_x": sx.to_dict(),
                "scaler_y": sy.to_dict(),
                "features": list(FAULTS) + list(CONDS),
                "targets": list(SIGNALS),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "best_epoch": int(best_state[1]),
            }
        ),
        encoding="utf-8",
    )
    print(f"\nSurrogate held out on {len(va)} samples (best epoch {best_state[1]}):")
    print(f"  {'signal':<14} {'mean |err|':>11} {'p95 |err|':>11}   (noise sigmas)")
    for j, s in enumerate(SIGNALS):
        print(f"  {s:<14} {err[:, j].mean():>11.2f} {np.percentile(err[:, j], 95):>11.2f}")
    print("\nA surrogate worth inverting through should sit near or below 1 sigma:")
    print("below the noise, the inverse solve is limited by the sensors, not by")
    print("the stand-in for the physics.")
    print(f"\n-> {ARTIFACTS / 'pinn_surrogate.json'}")


def load_surrogate() -> tuple[MLP, Scaler, Scaler]:
    d = json.loads((ARTIFACTS / "pinn_surrogate.json").read_text(encoding="utf-8"))
    return MLP.from_dict(d["net"]), Scaler.from_dict(d["scaler_x"]), Scaler.from_dict(d["scaler_y"])


# --------------------------------------------------------------------------
# Stage 2: the inverse network, trained on the physics residual
# --------------------------------------------------------------------------


def fleet_observations(limit: int = 60000, seed: int = 5) -> pd.DataFrame:
    """Real telemetry to invert: observed signals plus known conditions."""
    import duckdb

    db = ROOT / "data" / "servicesense.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    sql = f"""
    SELECT * FROM (
        SELECT t.asset_id, t.ts,
               t.suction_p, t.discharge_p, t.superheat, t.subcool,
               t.ambient_t_max AS ambient_c, t.load_frac,
               CASE WHEN a.refrigerant = 'R410A' THEN 1.0 ELSE 0.0 END AS is_r410a,
               a.asset_type,
               1.0 - t.health_condenser_coil AS true_cond_deg,
               1.0 - t.health_refrigerant_circuit AS true_charge_loss,
               1.0 - t.health_evaporator AS true_evap_deg
        FROM telemetry_daily t JOIN assets a USING (asset_id)
        WHERE t.suction_p IS NOT NULL
    ) USING SAMPLE {int(limit)} ROWS (reservoir, {int(seed)})
    """
    return con.execute(sql).df()


def train_inverse(epochs: int = 260, hidden: int = 48, lam: float = 0.02, seed: int = 9) -> None:
    """Fit F with the physics residual. No labels are used anywhere here."""
    surro, sx, sy = load_surrogate()
    df = add_cond_columns(fleet_observations())
    print(f"inverting {len(df)} real asset-days (no labels used in training)")

    obs = df[list(SIGNALS)].to_numpy(dtype=np.float64)
    cond = df[list(CONDS)].to_numpy(dtype=np.float64)

    # F sees the observations and the conditions, both standardised.
    f_in = np.hstack([obs, cond])
    sf = Scaler(f_in)
    fin = sf(f_in)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(fin))
    n_val = max(int(0.15 * len(fin)), 1)
    va, tr = idx[:n_val], idx[n_val:]

    net = MLP([fin.shape[1], hidden, hidden, len(FAULTS)], seed=seed,
              out_lo=FAULT_LO, out_hi=FAULT_HI)

    # Pre-scaled constants so the inner loop does no redundant work
    sx_mu, sx_sd = sx.mu, sx.sd
    sy_mu, sy_sd = sy.mu, sy.sd

    def physics_loss(rows: np.ndarray) -> tuple[float, float]:
        p = net(fin[rows])
        s_in = (np.hstack([p, cond[rows]]) - sx_mu) / sx_sd
        pred = sy.inverse(surro(s_in))
        resid = ((pred - obs[rows]) / SIGMA) ** 2
        prior = ((p - FAULT_HEALTHY) / (FAULT_HI - FAULT_LO)) ** 2
        return float(resid.mean()), float(prior.mean())

    batch = 512
    for ep in range(epochs):
        order = rng.permutation(len(tr))
        lr = 2e-3 * (0.5 ** (ep / 100))
        for k in range(0, len(order), batch):
            b = tr[order[k : k + batch]]
            nb = len(b)

            # forward: F then the frozen surrogate
            p, f_cache = net.forward(fin[b])
            s_in = (np.hstack([p, cond[b]]) - sx_mu) / sx_sd
            pred_std, s_cache = surro.forward(s_in)
            pred = pred_std * sy_sd + sy_mu

            # d(residual)/d(prediction), then back through the surrogate
            w = 1.0 / (SIGMA**2)
            d_pred = 2.0 * (pred - obs[b]) * w / (nb * len(SIGNALS))
            _, d_s_in = surro.backward(s_cache, d_pred * sy_sd, want_dx=True)
            # Out of the surrogate's input layer, only the fault columns
            # feed back into F; the condition columns are given, not learned.
            d_p = d_s_in[:, : len(FAULTS)] / sx_sd[: len(FAULTS)]

            # the healthy prior
            span = FAULT_HI - FAULT_LO
            d_p = d_p + lam * 2.0 * (p - FAULT_HEALTHY) / span**2 / (nb * len(FAULTS))

            grads, _ = net.backward(f_cache, d_p)
            net.step(grads, lr)

        if ep % 30 == 0 or ep == epochs - 1:
            r_tr, pr_tr = physics_loss(tr[:8000])
            r_va, _ = physics_loss(va)
            print(
                f"  epoch {ep:4d}  physics residual (sigma^2): train {r_tr:8.3f}  "
                f"val {r_va:8.3f}   prior {pr_tr:.4f}",
                flush=True,
            )

    (ARTIFACTS / "pinn_inverse.json").write_text(
        json.dumps(
            {
                "net": net.to_dict(),
                "scaler_in": sf.to_dict(),
                "inputs": list(SIGNALS) + list(CONDS),
                "outputs": list(FAULTS),
                "lambda_prior": lam,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
            }
        ),
        encoding="utf-8",
    )
    r_va, _ = physics_loss(va)
    print(f"\nHeld-out physics residual: {r_va:.3f} (in squared noise sigmas)")
    print("1.0 means the inferred state reproduces the sensors to within the")
    print("noise floor - the best any fault state could do.")
    print(f"\n-> {ARTIFACTS / 'pinn_inverse.json'}")


def load_inverse() -> tuple[MLP, Scaler]:
    d = json.loads((ARTIFACTS / "pinn_inverse.json").read_text(encoding="utf-8"))
    return MLP.from_dict(d["net"]), Scaler.from_dict(d["scaler_in"])


def infer(df: pd.DataFrame) -> np.ndarray:
    """Inferred fault state for rows carrying SIGNALS and CONDS."""
    net, sf = load_inverse()
    f_in = np.hstack(
        [df[list(SIGNALS)].to_numpy(dtype=np.float64), df[list(CONDS)].to_numpy(dtype=np.float64)]
    )
    return net(sf(f_in))


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def gradcheck(seed: int = 0) -> None:
    """Analytic gradients against central finite differences.

    Hand-written backprop is exactly the kind of code that trains to a
    plausible-looking loss while computing the wrong derivative, so this is
    not optional. It checks the scaled-sigmoid output head and the
    input-gradient path too, since stage 2 depends on both.
    """
    rng = np.random.default_rng(seed)
    ok = True
    for label, net, want_dx in [
        ("linear head", MLP([5, 7, 6, 3], seed=1), True),
        ("sigmoid head", MLP([5, 7, 6, 3], seed=2, out_lo=FAULT_LO, out_hi=FAULT_HI), True),
    ]:
        x = rng.normal(size=(11, 5))
        target = rng.normal(size=(11, 3))

        def loss_of(net=net, x=x, target=target) -> float:
            return float(((net(x) - target) ** 2).mean())

        pred, cache = net.forward(x)
        d = 2.0 * (pred - target) / pred.size
        grads, dx = net.backward(cache, d, want_dx=want_dx)

        # parameters
        worst_p = 0.0
        for gi, p in enumerate(net.params()):
            flat = p.ravel()
            gflat = grads[gi].ravel()
            for k in rng.choice(flat.size, size=min(12, flat.size), replace=False):
                orig = flat[k]
                h = 1e-6 * max(abs(orig), 1.0)
                flat[k] = orig + h
                lp = loss_of()
                flat[k] = orig - h
                lm = loss_of()
                flat[k] = orig
                num = (lp - lm) / (2 * h)
                den = max(abs(num), abs(gflat[k]), 1e-10)
                worst_p = max(worst_p, abs(num - gflat[k]) / den)

        # input gradient
        worst_x = 0.0
        if want_dx:
            for _ in range(15):
                i = rng.integers(x.shape[0])
                j = rng.integers(x.shape[1])
                orig = x[i, j]
                h = 1e-6
                x[i, j] = orig + h
                lp = loss_of()
                x[i, j] = orig - h
                lm = loss_of()
                x[i, j] = orig
                num = (lp - lm) / (2 * h)
                den = max(abs(num), abs(dx[i, j]), 1e-10)
                worst_x = max(worst_x, abs(num - dx[i, j]) / den)

        good = worst_p < 1e-5 and worst_x < 1e-5
        ok = ok and good
        print(
            f"  [{'PASS' if good else 'FAIL'}] {label:<14} "
            f"max rel err: params {worst_p:.2e}  input {worst_x:.2e}"
        )
    print(f"\nGRADCHECK: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def evaluate() -> None:
    """Does the inferred fault state track the damage that actually exists?

    This is the test the twin residual failed. The generator knows each
    component's true health, which no field dataset would, so it can be used
    here as a yardstick - it is never used in training.
    """
    df = add_cond_columns(fleet_observations(limit=40000, seed=23))
    p = infer(df)
    inferred = pd.DataFrame(p, columns=list(FAULTS))

    truth = {
        "cond_deg": df["true_cond_deg"].to_numpy(),
        "charge_frac": df["true_charge_loss"].to_numpy(),  # higher loss = lower charge
        "evap_deg": df["true_evap_deg"].to_numpy(),
    }
    amb = df["ambient_c"].to_numpy()

    def pair(v: np.ndarray, t: np.ndarray) -> tuple[float, int]:
        """Correlate on the rows where the yardstick exists.

        health_evaporator is only populated for chillers - the other families
        have no evaporator component in the BOM - so a whole-fleet correlation
        against it returns nan rather than a weak number.
        """
        m = np.isfinite(v) & np.isfinite(t)
        if m.sum() < 30 or np.std(t[m]) < 1e-12:
            return float("nan"), int(m.sum())
        return float(np.corrcoef(v[m], t[m])[0, 1]), int(m.sum())

    print(f"\nn = {len(df)} held-out asset-days\n")
    print(f"{'inferred':<14} {'vs own true damage':>20} {'n':>8} {'vs ambient':>12}")
    print("-" * 56)
    for f in FAULTS:
        v = inferred[f].to_numpy()
        r, n = pair(v, truth[f])
        # charge_frac falls as the leak grows, so flip it for comparison
        if f == "charge_frac" and np.isfinite(r):
            r = -r
        ra, _ = pair(v, amb)
        print(f"{f:<14} {r:>20.3f} {n:>8d} {ra:>12.3f}")

    # Is the inverse pinned against its own bounds? That is the signature of
    # a misspecified forward model: no reachable state fits, so the optimiser
    # runs to the edge of the feasible set and stops there.
    print("\nFraction of assets pinned at a parameter bound:\n")
    for j, f in enumerate(FAULTS):
        v = inferred[f].to_numpy()
        span = FAULT_HI[j] - FAULT_LO[j]
        at_lo = float(np.mean(v < FAULT_LO[j] + 0.02 * span))
        at_hi = float(np.mean(v > FAULT_HI[j] - 0.02 * span))
        print(f"  {f:<14} at lower bound {at_lo:6.1%}   at upper bound {at_hi:6.1%}")

    print("\nSign convention: a positive correlation means the inferred state")
    print("rises with the damage it is meant to measure (charge_frac is")
    print("flipped, since charge falls as a leak grows).")

    # Cross-correlation: does cond_deg pick up condenser damage specifically,
    # or is it just responding to overall degradation?
    print("\nCross-talk - each inferred parameter against every true damage:\n")
    print(f"{'':<14}" + "".join(f"{'true ' + k[:9]:>18}" for k in truth))
    for f in FAULTS:
        v = inferred[f].to_numpy()
        cells = []
        for t in truth.values():
            r, _ = pair(v, t)
            if f == "charge_frac" and np.isfinite(r):
                r = -r
            cells.append(f"{r:>18.3f}")
        print(f"{f:<14}" + "".join(cells))
    print("\nThe diagonal should dominate its row. Where it does not, the")
    print("parameter is reporting general degradation rather than its own mode.")

    # Where does the physics stop explaining the data? If the forward model
    # is sound and only the fault state is unknown, the residual should sit
    # near the noise floor (1.0) everywhere. It does not: it is around 100
    # across the fleet and still 30 for healthy machines at the calibration
    # ambient, so no reachable fault state reproduces these sensors. The
    # forward model itself disagrees with the data, which is not something a
    # better optimiser fixes.
    #
    # The split below is reported as measured rather than as expected: the
    # residual does not grow monotonically with distance from the calibration
    # ambient, so ambient alone is not the whole story - the load response
    # mismatch contributes as well.
    surro, sx, sy = load_surrogate()
    s_in = sx(np.hstack([p, df[list(CONDS)].to_numpy(dtype=np.float64)]))
    pred = sy.inverse(surro(s_in))
    resid = (((pred - df[list(SIGNALS)].to_numpy(dtype=np.float64)) / SIGMA) ** 2).mean(axis=1)

    print("\nPhysics residual vs distance from the calibration ambient (30 C),")
    print("in squared noise sigmas. 1.0 would mean a perfect fit:\n")
    d_cal = np.abs(amb - 30.0)
    bins = [0, 2, 5, 8, 12, 100]
    print(f"  {'|ambient - 30C|':<18} {'n':>7} {'median':>9} {'mean':>9}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d_cal >= lo) & (d_cal < hi)
        if m.sum() < 20:
            continue
        lbl = f"{lo}-{hi} K" if hi < 100 else f"{lo}+ K"
        print(
            f"  {lbl:<18} {int(m.sum()):>7d} {np.median(resid[m]):>9.1f} {resid[m].mean():>9.1f}"
        )

    healthy = (df["true_cond_deg"].to_numpy() < 0.1) & (df["true_charge_loss"].to_numpy() < 0.1)
    near = healthy & (d_cal < 3.0)
    if near.sum() > 20:
        print(
            f"\n  Healthy machines within 3 K of the calibration point "
            f"(n={int(near.sum())}): median residual {np.median(resid[near]):.1f}"
        )

    print("\nDistribution of the inferred state:\n")
    print(f"{'':<14} {'mean':>8} {'p05':>8} {'p50':>8} {'p95':>8}")
    for f in FAULTS:
        v = inferred[f].to_numpy()
        print(
            f"{f:<14} {v.mean():>8.3f} {np.percentile(v, 5):>8.3f} "
            f"{np.percentile(v, 50):>8.3f} {np.percentile(v, 95):>8.3f}"
        )


def infer_fleet(out_name: str = "fault_state.parquet") -> None:
    """Write the latest inferred fault state per asset, for the API and UI."""
    import duckdb

    con = duckdb.connect(str(ROOT / "data" / "servicesense.duckdb"), read_only=True)
    df = con.execute(
        """
        WITH latest AS (
            SELECT asset_id, MAX(ts) AS ts FROM telemetry_daily
            WHERE suction_p IS NOT NULL GROUP BY asset_id
        )
        SELECT t.asset_id, t.ts,
               t.suction_p, t.discharge_p, t.superheat, t.subcool,
               t.ambient_t_max AS ambient_c, t.load_frac,
               CASE WHEN a.refrigerant = 'R410A' THEN 1.0 ELSE 0.0 END AS is_r410a,
               a.asset_type, a.capacity_kw, a.refrigerant
        FROM telemetry_daily t
        JOIN latest USING (asset_id, ts)
        JOIN assets a USING (asset_id)
        """
    ).df()
    df = add_cond_columns(df)
    p = infer(df)
    out = df[["asset_id", "ts", "asset_type", "capacity_kw", "refrigerant"]].copy()
    for j, f in enumerate(FAULTS):
        out[f] = p[:, j]

    # Report the physics residual per asset so the UI can say how well the
    # inferred state actually explains that machine's sensors.
    surro, sx, sy = load_surrogate()
    s_in = sx(np.hstack([p, df[list(CONDS)].to_numpy(dtype=np.float64)]))
    pred = sy.inverse(surro(s_in))
    resid = (((pred - df[list(SIGNALS)].to_numpy(dtype=np.float64)) / SIGMA) ** 2).mean(axis=1)
    out["physics_residual"] = resid
    out["explains_sensors"] = resid < 4.0  # within 2 sigma per signal
    for f in FAULTS:
        out[f"{f}_validated"] = FAULT_VALIDATED[f]

    DATA.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA / out_name, index=False)
    print(f"{len(out)} assets -> {DATA / out_name}")
    print(f"  mean physics residual {resid.mean():.2f} sigma^2; "
          f"{int(out['explains_sensors'].sum())} of {len(out)} within 2 sigma per signal")
    for f in FAULTS:
        print(f"  {f:<14} mean {out[f].mean():.3f}  p95 {np.percentile(out[f], 95):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Physics-informed fault-state inference")
    ap.add_argument("--identifiability", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--train-surrogate", action="store_true")
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--train-inverse", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--epochs", type=int, default=0)
    a = ap.parse_args()

    did = False
    if a.identifiability:
        identifiability()
        did = True
    if a.gradcheck:
        gradcheck()
        did = True
    if a.sample:
        sample_twin(n=a.n)
        did = True
    if a.train_surrogate:
        train_surrogate(**({"epochs": a.epochs} if a.epochs else {}))
        did = True
    if a.train_inverse:
        train_inverse(**({"epochs": a.epochs} if a.epochs else {}))
        did = True
    if a.evaluate:
        evaluate()
        did = True
    if a.infer:
        infer_fleet()
        did = True
    if not did:
        ap.print_help()


if __name__ == "__main__":
    main()
