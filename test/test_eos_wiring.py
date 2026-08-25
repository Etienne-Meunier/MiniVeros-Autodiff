#!/usr/bin/env python3
"""
Wiring smoke test for eq_of_state_type dispatch: runs acc_basic with the
type overridden after setup -- same "swap the static config after build"
trick test_pressure_solver.py uses for enable_streamfunction -- and compares
against real veros doing the identical override, for each of the newly
ported types (1, 2, 4). The pure functions themselves are already verified
bit-exact (test_eos_types.py); this instead exercises the *dispatch*
(core/thermodynamics.py's calc_eq_of_state call site) inside a real stepping
loop, which is what caught the missing int_drhodT broadcast (see ISSUES.md).

psi is checked scale-normalized rather than by plain tolerance, same
reasoning as test_all_setups.py: it carries an arbitrary per-island gauge
and has cells near zero in both implementations independently.

Usage:
    python test/test_eos_wiring.py [--steps N] [--veros-path PATH]
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from util import configure_veros_runtime, compare_field

DEFAULT_VEROS_PATH = REPO_ROOT.parent / "veros"
EOS_TYPES = (1, 2, 4)


def build_mini(n_steps, eos_type):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import jax

    jax.config.update("jax_enable_x64", True)
    from mini_veros import loop
    from mini_veros.setups.acc import basic as acc_basic

    model, step0, forcing_fn = acc_basic.build()
    model = dataclasses.replace(model, config=dataclasses.replace(model.config, eq_of_state_type=eos_type))

    step_jit = jax.jit(lambda s: loop.step(model, s, forcing_fn))
    step = step0
    for _ in range(n_steps):
        step = step_jit(step)
    return {name: np.asarray(getattr(step.state, name)) for name in ("u", "v", "temp", "salt", "psi", "tke")}


def build_real(n_steps, eos_type):
    from veros.setups.acc_basic.acc_basic import ACCBasicSetup
    from veros.routines import veros_routine
    from veros.state import VerosSettings

    class NoIOSetup(ACCBasicSetup):
        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

    sim = NoIOSetup(override=dict(runlen=0))
    sim.setup()
    with VerosSettings.unlock(sim.state.settings):
        sim.state.settings.eq_of_state_type = eos_type

    for _ in range(n_steps):
        sim.step(sim.state)

    vs = sim.state.variables
    return {name: np.asarray(getattr(vs, name)[..., vs.tau]) for name in ("u", "v", "temp", "salt", "psi", "tke")}


def check_type(eos_type, n_steps):
    mini = build_mini(n_steps, eos_type)
    real = build_real(n_steps, eos_type)

    ok = True
    for name in sorted(mini):
        if name == "psi":
            scale = np.max(np.abs(real[name]))
            scale = scale if scale > 0 else 1.0
            normalized = np.max(np.abs(mini[name] - real[name])) / scale
            passed = normalized < 1e-6
            print(f"  eq_of_state_type={eos_type}  psi   scale-normalized error={normalized:.3e} {'ok' if passed else 'FAIL'}")
        else:
            r = compare_field(name, mini[name], real[name], atol=1e-6, rtol=1e-6)
            passed = r.get("ok", False)
            print(f"  eq_of_state_type={eos_type}  {name:6s}max_abs={r.get('max_abs', float('nan')):.3e} max_rel={r.get('max_rel', float('nan')):.3e} {'ok' if passed else 'FAIL'}")
        ok = ok and passed
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--veros-path", type=Path, default=DEFAULT_VEROS_PATH)
    args = parser.parse_args()

    if not (args.veros_path / "veros" / "__init__.py").exists():
        parser.error(f"no veros package found at {args.veros_path}")

    configure_veros_runtime(args.veros_path)

    all_ok = True
    for eos_type in EOS_TYPES:
        print(f"--- acc_basic under eq_of_state_type={eos_type}, {args.steps} steps ---")
        all_ok = check_type(eos_type, args.steps) and all_ok

    print()
    print("PASS -- eq_of_state_type dispatch matches veros in a live rollout for types 1, 2, 4" if all_ok else "FAIL -- see mismatches above")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
