#!/usr/bin/env python3
"""
Same idea as generate_report_data.py, but for the surface-pressure external
mode (enable_streamfunction=False, see test_pressure_solver.py) -- records
error evolution over a longer horizon for the trust report's pressure-solver
section. Saves test/results/pressure_acc_basic.npz.

Usage:
    python test/generate_pressure_report_data.py [--steps N] [--veros-path PATH]
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from util import configure_veros_runtime, compute_error_evolution

DEFAULT_VEROS_PATH = REPO_ROOT.parent / "veros"
RESULTS_DIR = REPO_ROOT / "test" / "results"
FIELDS = ("u", "v", "temp", "salt", "tke", "psi")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--record-interval", type=int, default=10)
    parser.add_argument("--veros-path", type=Path, default=DEFAULT_VEROS_PATH)
    args = parser.parse_args()

    if not (args.veros_path / "veros" / "__init__.py").exists():
        parser.error(f"no veros package found at {args.veros_path}")

    configure_veros_runtime(args.veros_path)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import jax

    jax.config.update("jax_enable_x64", True)
    from mini_veros import loop
    from mini_veros.setups.acc import basic as acc_basic

    model, step0, forcing_fn = acc_basic.build()
    model = dataclasses.replace(model, config=dataclasses.replace(model.config, enable_streamfunction=False))
    step_jit = jax.jit(lambda s: loop.step(model, s, forcing_fn))

    from veros.setups.acc_basic.acc_basic import ACCBasicSetup
    from veros.routines import veros_routine

    class NoIOSetup(ACCBasicSetup):
        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

    sim = NoIOSetup(override=dict(runlen=0, enable_streamfunction=False))
    sim.setup()

    timesteps = [0]
    mini_states = [{f: np.asarray(getattr(step0.state, f)) for f in FIELDS}]
    vs = sim.state.variables
    real_states = [{f: np.asarray(getattr(vs, f)[..., vs.tau]) for f in FIELDS}]

    step = step0
    for i in range(1, args.steps + 1):
        step = step_jit(step)
        sim.step(sim.state)
        if i % args.record_interval == 0:
            timesteps.append(i)
            mini_states.append({f: np.asarray(getattr(step.state, f)) for f in FIELDS})
            vs = sim.state.variables
            real_states.append({f: np.asarray(getattr(vs, f)[..., vs.tau]) for f in FIELDS})

    errors = compute_error_evolution(timesteps, mini_states, real_states)

    out = dict(timesteps=np.asarray(timesteps))
    for field, data in errors.items():
        for key in ("max_abs_errors", "mean_abs_errors"):
            out[f"err_{field}_{key}"] = np.asarray(data[key])
    out["psi_real_scale"] = np.max(np.abs(real_states[-1]["psi"]))

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "pressure_acc_basic.npz"
    np.savez(out_path, **out)
    print("saved", out_path)


if __name__ == "__main__":
    main()
