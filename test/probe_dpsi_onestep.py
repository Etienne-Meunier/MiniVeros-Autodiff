#!/usr/bin/env python3
"""
One-step probe: run both mini_veros and real veros for N_WARMUP steps (to
build up non-trivial dpsi history), then for exactly ONE more step, extract
each side's `forc` (elliptic solve RHS), `x0` (warm-start initial guess),
and solved `dpsi` -- and diff them directly. Answers: are forc/x0 actually
equal between the two implementations, and if so, does the solve itself
produce a different dpsi?

Real veros's side is captured by monkeypatching JAXSciPySolver.solve and
letting a NORMAL, unmodified sim.step() run through it -- NOT by calling
prepare_forcing standalone, which skips the coriolis/advection/friction
chain that freshly overwrites vs.du[...,tau] each step (see conversation:
an earlier version of this script called prepare_forcing directly and was
silently wrong because of this).

Usage:
    python test/probe_dpsi_onestep.py [N_WARMUP]
"""

import sys
import importlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))
from util import configure_veros_runtime  # noqa: E402

VEROS_PATH = REPO_ROOT.parent / "veros"
N_WARMUP = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def report(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        print(f"  {name:6s}  SHAPE MISMATCH mini={a.shape} real={b.shape}")
        return
    diff = np.abs(a - b)
    print(
        f"  {name:6s}  max_abs_diff={diff.max():.6e}  "
        f"mini_scale={np.abs(a).max():.6e}  real_scale={np.abs(b).max():.6e}"
    )


def main():
    configure_veros_runtime(VEROS_PATH)
    sys.path.insert(0, str(REPO_ROOT / "src"))

    # ---------------- mini_veros ----------------
    from mini_veros import loop
    from mini_veros.core.external.solvers import scipy_jax as mini_scipy_jax

    setup_mod = importlib.import_module("mini_veros.setups.acc.basic")
    model, mini_step, forcing_fn = setup_mod.build()

    for _ in range(N_WARMUP):
        mini_step = loop.step(model, mini_step, forcing_fn(model, mini_step.state))

    captured_mini = {}
    orig_solve = mini_scipy_jax.elliptic_solve

    def capture_mini(model_, rhs, x0, boundary_val=None):
        captured_mini.setdefault("forc", np.asarray(rhs))
        captured_mini.setdefault("x0", np.asarray(x0))
        result = orig_solve(model_, rhs, x0, boundary_val=boundary_val)
        captured_mini.setdefault("dpsi", np.asarray(result))
        return result

    mini_scipy_jax.elliptic_solve = capture_mini
    try:
        loop.step(model, mini_step, forcing_fn(model, mini_step.state))
    finally:
        mini_scipy_jax.elliptic_solve = orig_solve

    # ---------------- real veros ----------------
    from veros.routines import veros_routine
    from veros.core.external.solvers.scipy_jax import JAXSciPySolver

    real_setup_mod = importlib.import_module("veros.setups.acc_basic.acc_basic")
    RealSetupClass = getattr(real_setup_mod, "ACCBasicSetup")

    class NoIOSetup(RealSetupClass):
        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

    sim = NoIOSetup(override=dict(runlen=0))
    sim.setup()

    for _ in range(N_WARMUP):
        sim.step(sim.state)

    captured_real = {}
    orig_real_solve = JAXSciPySolver.solve

    def capture_real(self, state, rhs, x0, boundary_val=None):
        captured_real.setdefault("forc", np.asarray(rhs))
        captured_real.setdefault("x0", np.asarray(x0))
        result = orig_real_solve(self, state, rhs, x0, boundary_val=boundary_val)
        captured_real.setdefault("dpsi", np.asarray(result))
        return result

    JAXSciPySolver.solve = capture_real
    try:
        sim.step(sim.state)  # ONE normal, unmodified step -- solve_streamfunction runs inside this
    finally:
        JAXSciPySolver.solve = orig_real_solve

    # ---------------- compare ----------------
    print(f"=== one-step probe after {N_WARMUP} warmup step(s) ===")
    report("forc", captured_mini["forc"], captured_real["forc"])
    report("x0", captured_mini["x0"], captured_real["x0"])
    report("dpsi", captured_mini["dpsi"], captured_real["dpsi"])


if __name__ == "__main__":
    main()
