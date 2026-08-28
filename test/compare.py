#!/usr/bin/env python3
"""
1:1 numerical comparison between this repo (mini_veros) and upstream veros.

Both are built from the *same* setup (identical grid, topology, parameters,
forcing -- acc_basic.py/acc.py's docstrings promise "every number here is
copied straight from the real setup file, no simplified version"), so they
should start from an identical initial state and stay identical for as long
as mini_veros's port is faithful. This script:

  1. builds both models
  2. diffs the grid/topology/boundary-condition arrays (should match to
     ~machine precision -- a mismatch here means the *setups* diverge, not
     the physics)
  3. diffs the initial prognostic state (u, v, temp, salt, psi, tke, and eke
     for --setup acc)
  4. runs N step()s in each and diffs the resulting state

Any field that drifts beyond tolerance points at a real port discrepancy
worth chasing down (see ../ISSUES.md for known gaps).

Usage:
    python test/compare.py [--setup acc_basic|acc] [--steps N] [--veros-path PATH] [--atol A] [--rtol R]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# mini_veros setup names that don't map 1:1 to a "mini_veros.setups.<name>" module path
_MINI_SETUP_MODULE = {
    "acc_basic": "mini_veros.setups.acc.basic",
    "acc": "mini_veros.setups.acc.full",
    "global_4deg": "mini_veros.setups.global_4deg.default",
}
DEFAULT_VEROS_PATH = REPO_ROOT.parent / "veros"

# prognostic fields mini_veros actually evolves. eke only for setups with
# enable_eke=True (acc, global_4deg): acc_basic has enable_eke=False, and
# real veros raises on accessing an inactive variable ("eke is not active in
# this configuration") rather than returning zeros, so it can't be in the
# common list.
STATE_FIELDS = ["u", "v", "temp", "salt", "psi", "tke"]
EKE_SETUPS = ("acc", "global_4deg")

# static grid/topology fields -- same names in mini_veros's Grid/
# BoundaryConditions and in real veros's state.variables
GRID_FIELDS = [
    "xt", "xu", "yt", "yu", "zt", "zw",
    "dxt", "dxu", "dyt", "dyu", "dzt", "dzw",
    "cost", "cosu", "tantr", "area_t", "area_u", "area_v",
    "coriolis_t", "beta",
]
BC_FIELDS = ["kbot", "maskT", "maskU", "maskV", "maskW", "maskZ", "ht", "hu", "hv", "hur", "hvr"]

# real veros setup class name per --setup choice, all under veros.setups.<name>.<name>
REAL_SETUP_CLASS = {"acc_basic": "ACCBasicSetup", "acc": "ACCSetup", "global_4deg": "GlobalFourDegreeSetup"}


def configure_real_veros_runtime(veros_path):
    """
    Must run before ANY veros.core-importing code, including build_mini
    below (global_4deg imports veros.tools for asset download/cache, see
    its module docstring -- veros.tools transitively imports
    veros.core.operators). veros's runtime_settings lock themselves
    (rs.__locked__ = True) the moment veros.core is first imported
    (veros/core/__init__.py) -- any rs.* assignment after that point raises
    "Runtime settings cannot be modified after import of core modules",
    even to reassign the same value. So this has to run exactly once, here,
    before build_mini's import of a veros.tools-using setup module.
    """
    sys.path.insert(0, str(veros_path))

    from veros import runtime_settings as rs

    rs.backend = "jax"
    rs.device = "cpu"
    rs.float_type = "float64"
    # mini_veros permanently uses the pure-JAX bicgstab elliptic solver (see
    # ISSUES.md); real veros's "best" solver only picks that backend on
    # GPU+float64, otherwise falling back to scipy (host-side, untraced).
    # Force the same backend here so a mismatch reflects the *port*, not an
    # incidental solver choice.
    rs.linear_solver = "scipy_jax"

    import jax
    jax.config.update("jax_enable_x64", True)


def build_mini(setup_name, n_steps, veros_path):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    # global_4deg imports veros.tools (asset download/cache only, not
    # physics) at module load time, so the real veros repo needs to be
    # importable here too, not just in build_real below.
    sys.path.insert(0, str(veros_path))

    import jax
    jax.config.update("jax_enable_x64", True)

    from mini_veros import loop
    import importlib
    setup_mod = importlib.import_module(_MINI_SETUP_MODULE.get(setup_name, f"mini_veros.setups.{setup_name}"))

    model, step0, forcing_fn = setup_mod.build()

    state_fields = STATE_FIELDS + (["eke"] if setup_name in EKE_SETUPS else [])

    grid = {name: np.asarray(getattr(model.grid, name)) for name in GRID_FIELDS}
    bc = {name: np.asarray(getattr(model.boundary_conditions, name)) for name in BC_FIELDS}
    s0 = {name: np.asarray(getattr(step0.state, name)) for name in state_fields}

    step = step0
    with jax.disable_jit():
        for _ in range(n_steps):
            step = loop.step(model, step, forcing_fn(model, step.state))

    s_final = {name: np.asarray(getattr(step.state, name)) for name in state_fields}

    return grid, bc, s0, s_final


def build_real(setup_name, veros_path, n_steps):
    # runtime_settings (backend/device/float_type/linear_solver) are
    # configured once in configure_real_veros_runtime, called before this in
    # main() -- see that function's docstring for why it can't happen here.
    sys.path.insert(0, str(veros_path))

    from veros.routines import veros_routine
    import importlib
    real_setup_mod = importlib.import_module(f"veros.setups.{setup_name}.{setup_name}")
    RealSetupClass = getattr(real_setup_mod, REAL_SETUP_CLASS[setup_name])

    class NoIOSetup(RealSetupClass):
        # never touch disk, regardless of --steps (real set_diagnostics only
        # configures *what* to write; clearing it disables all output/restart
        # writes -- same pattern veros's own pyom_consistency/acc_test.py uses)
        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

    sim = NoIOSetup(override=dict(runlen=0))
    sim.setup()

    vs = sim.state.variables
    state_fields = STATE_FIELDS + (["eke"] if setup_name in EKE_SETUPS else [])

    def field(name):
        return np.asarray(getattr(vs, name))

    grid = {name: field(name) for name in GRID_FIELDS}
    bc = {name: field(name) for name in BC_FIELDS}
    s0 = {name: field(name)[..., vs.tau] for name in state_fields}

    import jax

    # only the step loop runs under disable_jit -- veros's only jax.jit site is
    # routines.py's veros_kernel wrapper (routines.py:317), so this makes every
    # step()'s kernel call eager and a state mismatch below reflect the port
    # itself, not XLA fusion/optimization choices. setup() stays jitted: it's
    # deterministic grid/topology build, and island.py's isleperim deliberately
    # round-trips through numpy/scipy (land-mass labelling), which only survives
    # its onp-array being fed back into a veros_kernel because jax.jit's tracing
    # silently upcasts it -- disabling jit that early breaks on a bare
    # numpy.ndarray having no .at attribute.
    with jax.disable_jit():
        for _ in range(n_steps):
            sim.step(sim.state)

    vs = sim.state.variables
    s_final = {name: field(name)[..., vs.tau] for name in state_fields}

    return grid, bc, s0, s_final


def compare_field(name, a, b, atol, rtol):
    if a.shape != b.shape:
        return dict(name=name, ok=False, error=f"shape mismatch: mini={a.shape} real={b.shape}")

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom == 0, 1.0, denom)
    rel = diff / denom

    ok = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0
    max_rel = float(np.nanmax(rel)) if diff.size else 0.0
    argmax = np.unravel_index(np.nanargmax(diff), diff.shape) if diff.size else None

    return dict(
        name=name, ok=ok, shape=a.shape,
        max_abs=max_abs, max_rel=max_rel, argmax=argmax,
        mini_at_argmax=float(a[argmax]) if argmax is not None else None,
        real_at_argmax=float(b[argmax]) if argmax is not None else None,
    )


def print_section(title, mini_dict, real_dict, atol, rtol):
    print(f"\n=== {title} ===")
    all_ok = True
    for name in mini_dict:
        if name not in real_dict:
            print(f"  {name:12s}  SKIP (not found in reference)")
            continue
        r = compare_field(name, mini_dict[name], real_dict[name], atol, rtol)
        if "error" in r:
            print(f"  {name:12s}  FAIL  {r['error']}")
            all_ok = False
            continue
        status = "ok  " if r["ok"] else "FAIL"
        print(
            f"  {name:12s}  {status}  max_abs={r['max_abs']:.3e}  max_rel={r['max_rel']:.3e}"
            f"  shape={r['shape']}"
        )
        if not r["ok"]:
            print(
                f"               worst at {r['argmax']}: mini={r['mini_at_argmax']!r} "
                f"real={r['real_at_argmax']!r}"
            )
        all_ok = all_ok and r["ok"]
    return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--setup", choices=sorted(REAL_SETUP_CLASS), default="acc_basic",
        help="which setup to compare (default: acc_basic)",
    )
    parser.add_argument("--steps", type=int, default=5, help="number of step()s to run (default: 5)")
    parser.add_argument(
        "--veros-path", type=Path, default=DEFAULT_VEROS_PATH,
        help=f"path to the real veros repo (default: {DEFAULT_VEROS_PATH})",
    )
    # matches the elliptic solver's own convergence tolerance
    # (core/external/solvers/scipy_jax.py's bicgstab(..., atol=1e-8)) --
    # tighter than that flags noise from the solver itself, not a real
    # regression. Not calibrated per-field (e.g. psi's O(1e6) values still
    # fail this on an absolute basis) -- a real regression test with
    # per-field tolerances comes later, this is just a manual sanity check.
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-8)
    args = parser.parse_args()

    if not (args.veros_path / "veros" / "__init__.py").exists():
        parser.error(f"no veros package found at {args.veros_path} -- pass --veros-path")

    print(f"Running {args.setup} for {args.steps} step(s)...")
    print("  mini_veros:", REPO_ROOT)
    print("  veros:     ", args.veros_path)

    configure_real_veros_runtime(args.veros_path)

    mini_grid, mini_bc, mini_s0, mini_sf = build_mini(args.setup, args.steps, args.veros_path)
    real_grid, real_bc, real_s0, real_sf = build_real(args.setup, args.veros_path, args.steps)

    ok = True
    ok &= print_section("grid", mini_grid, real_grid, args.atol, args.rtol)
    ok &= print_section("boundary conditions / topology", mini_bc, real_bc, args.atol, args.rtol)
    ok &= print_section("initial state (t=0)", mini_s0, real_s0, args.atol, args.rtol)
    ok &= print_section(f"state after {args.steps} step(s)", mini_sf, real_sf, args.atol, args.rtol)

    print()
    if ok:
        print(f"PASS -- mini_veros matches veros within atol={args.atol:.1e} rtol={args.rtol:.1e}")
    else:
        print("FAIL -- see mismatches above")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
