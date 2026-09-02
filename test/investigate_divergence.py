#!/usr/bin/env python3
"""
Data generator for report/divergence_report.md -- the three experiments that
explain why mini_veros and veros trajectories separate on long runs.

Background: report/matrix_report.md shows 28/31 variants "FAIL" at 10950
steps while the three "ok" ones are 4-step snapshots. That is not 28 broken
ports. Three separate effects are stacked in that table, and each experiment
below isolates one.

  init   step-0 parity for every variant. Catches initial-condition
         mismatches, which no downstream physics agreement can undo. This is
         how the surface-pressure init bug was found (mini used to run an
         initial solve_pressure that real veros never runs).

  physics every variant, a few steps, with the solver's stopping rule
         tightened out of the way. What survives is the physics port itself.

  seed   the same variant run twice at the elliptic solver's default
         stopping rule (atol=1e-8, rtol=0 -- identical in both codes) and
         then at a tight one, recording per-step field differences. Shows
         that the first-step disagreement is set by the solver's tolerance,
         about seven orders of magnitude above float64 roundoff, and drops
         to roundoff when the tolerance is tightened.

  twin   real veros against itself from an initial condition perturbed by a
         relative `--perturb` (default 1e-12). No mini_veros involved. Gives
         the flow's own error-growth curve and saturation plateau, which is
         the yardstick the mini-vs-veros curve has to be read against.

  long   mini vs veros over the matrix's full horizon at a forced solver
         tolerance. Pairs with the matrix run's own (default-tolerance)
         curve: a smaller seed should delay the growth, not remove it.

  solver real veros against real veros with nothing changed but the linear
         solver backend (`scipy_jax` vs `scipy`, both veros's own supported
         options for identical physics). This is the control the `twin`
         experiment cannot be: two implementations of the same equations
         whose arithmetic differs at every step, rather than one trajectory
         kicked once at t=0. Runs each backend in its own subprocess, since
         veros locks its runtime settings at import.

Results go to $STORE/MiniVeros-Autodiff/results/divergence/, one .npz per
experiment; plot_divergence_report.py turns them into figures + the report.

Usage:
    python test/investigate_divergence.py init
    python test/investigate_divergence.py seed --variant acc_basic
    python test/investigate_divergence.py twin --variant acc_basic --perturb 1e-12
    python test/investigate_divergence.py long   --variant acc_basic
    python test/investigate_divergence.py solver --variant acc_basic
    python test/investigate_divergence.py all    --variant acc_basic

Then: python test/plot_divergence_report.py
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from setups_matrix import FAMILIES, VARIANTS, VARIANTS_BY_NAME
from util import configure_veros_runtime

DEFAULT_VEROS_PATH = REPO_ROOT / "veros"
STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
OUT_DIR = STORE / "MiniVeros-Autodiff" / "results" / "divergence"

# The elliptic solver's default stopping rule, shared by both codes:
# veros/core/external/solvers/scipy_jax.py and mini_veros's copy of it both
# call bicgstab(..., tol=0, atol=1e-8), i.e. an *absolute* residual bound.
DEFAULT_SOLVER_ATOL = 1e-8
TIGHT_SOLVER_ATOL = 1e-14


@contextlib.contextmanager
def forced_solver_atol(atol):
    """
    Force both implementations' bicgstab to `atol` for the duration.

    Both import the same jax.scipy.sparse.linalg.bicgstab: veros inside
    JAXSciPySolver.__init__ (so patching the module is enough), mini_veros
    at module scope in its solvers.scipy_jax (so that module's own attribute
    has to be patched too).
    """
    import jax.scipy.sparse.linalg as jssl

    from mini_veros.core.external.solvers import scipy_jax as mini_solver

    original = jssl.bicgstab

    def patched(A, b, x0=None, *, tol=0.0, atol=0.0, maxiter=None, M=None):
        return original(A, b, x0=x0, tol=0.0, atol=atol_forced, maxiter=maxiter, M=M)

    atol_forced = atol
    jssl.bicgstab = patched
    mini_solver_original = mini_solver.bicgstab
    mini_solver.bicgstab = patched
    try:
        yield
    finally:
        jssl.bicgstab = original
        mini_solver.bicgstab = mini_solver_original


def _scale(arr):
    """RMS of a reference field, used to turn absolute differences into a comparable number."""
    s = float(np.sqrt(np.nanmean(np.asarray(arr, dtype=np.float64) ** 2)))
    return s if s > 0 else 1.0


def _normalized_diffs(mini_state, real_state):
    """{field: max|mini-real| / rms(real)} for one recorded step."""
    out = {}
    for field in sorted(mini_state):
        a = np.asarray(mini_state[field], dtype=np.float64)
        b = np.asarray(real_state[field], dtype=np.float64)
        out[field] = float(np.nanmax(np.abs(a - b))) / _scale(b)
    return out


def experiment_init(veros_path, group=None):
    """Step-0 field-by-field parity between mini_veros and veros, for every variant."""
    from variant_util import build_mini_variant, build_real_variant

    selected = [v for v in VARIANTS if group is None or FAMILIES[v["family"]]["group"] == group]

    names, worst_field, worst_val = [], [], []
    print(f"{'variant':34s} {'worst field':>12s} {'max|diff|/rms':>14s}")
    for variant in selected:
        name, family, overrides = variant["name"], variant["family"], variant["overrides"]
        try:
            _, mini_s0, *_ = build_mini_variant(name, family, overrides, 0, veros_path)
            _, real_s0, *_ = build_real_variant(name, family, overrides, 0, veros_path)
        except Exception as exc:  # a variant real veros refuses to build shouldn't kill the sweep
            print(f"{name:34s} ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        diffs = _normalized_diffs(mini_s0, real_s0)
        field = max(diffs, key=diffs.get)
        names.append(name)
        worst_field.append(field)
        worst_val.append(diffs[field])
        print(f"{name:34s} {field:>12s} {diffs[field]:14.3e}", flush=True)

    return dict(
        variants=np.asarray(names),
        worst_field=np.asarray(worst_field),
        worst_normalized_diff=np.asarray(worst_val),
    )


def experiment_physics(veros_path, n_steps, atol, group=None):
    """
    Every variant, a few steps, with the solver's stopping rule tightened out
    of the way: what is left is the physics port itself.

    With the default atol=1e-8 this comparison is impossible -- the solver's
    own slack (about 1e-9 relative in psi, seven orders above float64
    roundoff) buries any real term-by-term difference.
    """
    from variant_util import build_mini_variant, build_real_variant

    selected = [v for v in VARIANTS if group is None or FAMILIES[v["family"]]["group"] == group]

    names, worst_field, worst_val = [], [], []
    print(f"forced solver atol={atol:g}, {n_steps} steps\n")
    print(f"{'variant':34s} {'worst field':>12s} {'max diff / rms':>15s}")
    for variant in selected:
        name, family, overrides = variant["name"], variant["family"], variant["overrides"]
        try:
            with forced_solver_atol(atol):
                _, _, _, _, _, mini_states = build_mini_variant(
                    name, family, overrides, n_steps, veros_path, n_steps
                )
                _, _, _, _, _, real_states = build_real_variant(
                    name, family, overrides, n_steps, veros_path, n_steps
                )
        except Exception as exc:
            print(f"{name:34s} ERROR {type(exc).__name__}: {exc}", flush=True)
            continue

        diffs = _normalized_diffs(mini_states[-1], real_states[-1])
        field = max(diffs, key=diffs.get)
        names.append(name)
        worst_field.append(field)
        worst_val.append(diffs[field])
        print(f"{name:34s} {field:>12s} {diffs[field]:15.3e}", flush=True)

    return dict(
        atol=np.asarray(atol),
        n_steps=np.asarray(n_steps),
        variants=np.asarray(names),
        worst_field=np.asarray(worst_field),
        worst_normalized_diff=np.asarray(worst_val),
    )


def experiment_seed(variant_name, veros_path, n_steps, atols):
    """Per-step mini-vs-veros differences at each solver tolerance in `atols`."""
    from variant_util import build_mini_variant, build_real_variant

    variant = VARIANTS_BY_NAME[variant_name]
    name, family, overrides = variant["name"], variant["family"], variant["overrides"]

    out = dict(variant=np.asarray(name), n_steps=np.asarray(n_steps), atols=np.asarray(atols))
    for atol in atols:
        with forced_solver_atol(atol):
            _, _, _, _, timesteps, mini_states = build_mini_variant(
                name, family, overrides, n_steps, veros_path, 1
            )
            _, _, _, _, _, real_states = build_real_variant(name, family, overrides, n_steps, veros_path, 1)

        per_step = [_normalized_diffs(m, r) for m, r in zip(mini_states, real_states)]
        fields = sorted(per_step[0])
        print(f"\n=== {name}: solver atol={atol:g} -- max|mini-real| / rms(real) ===")
        print("step " + "".join(f"{f:>12s}" for f in fields))
        for t, row in zip(timesteps, per_step):
            print(f"{t:4d} " + "".join(f"{row[f]:12.3e}" for f in fields), flush=True)

        out["timesteps"] = np.asarray(timesteps)
        tag = f"atol{atol:g}"
        for f in fields:
            out[f"{tag}_{f}"] = np.asarray([row[f] for row in per_step])

    return out


def experiment_twin(variant_name, veros_path, n_steps, record_interval, perturb, seeds=(1234,)):
    """
    Real veros vs real veros, second run started from temp * (1 + perturb * noise).

    Deliberately does not touch mini_veros: this measures the flow's own
    sensitivity to a small initial perturbation, which is what the
    mini-vs-veros error curve has to be compared against.
    """
    import importlib

    import jax.numpy as jnp
    from veros.routines import veros_routine

    from variant_util import _prognostic_fields

    variant = VARIANTS_BY_NAME[variant_name]
    name, family, overrides = variant["name"], variant["family"], variant["overrides"]
    spec = FAMILIES[family]
    RealSetupClass = getattr(importlib.import_module(spec["real_module"]), spec["real_class"])

    class NoIOSetup(RealSetupClass):
        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

    def run(seed):
        sim = NoIOSetup(override=dict(runlen=0, **overrides))
        sim.setup()
        vs = sim.state.variables
        prog = _prognostic_fields(sim.state.settings.enable_eke, sim.state.settings.enable_tke)

        if seed is not None:
            rng = np.random.default_rng(seed)
            temp = np.array(vs.temp, dtype=np.float64, copy=True)
            noise = rng.standard_normal(temp.shape[:3])
            # every leapfrog time level, so the start is self-consistent
            for level in range(temp.shape[3]):
                temp[..., level] = temp[..., level] * (1.0 + perturb * noise)
            with sim.state.variables.unlock():
                vs.temp = jnp.asarray(temp)

        def snapshot():
            v = sim.state.variables
            return {n: np.asarray(getattr(v, n)[..., v.tau]) for n in prog}

        timesteps, states = [0], [snapshot()]
        for i in range(n_steps):
            sim.step(sim.state)
            if (i + 1) % record_interval == 0:
                timesteps.append(i + 1)
                states.append(snapshot())
        return prog, timesteps, states

    prog, timesteps, base = run(None)
    print("twin: baseline done", flush=True)

    # one curve per seed: a single realization says nothing about how much of
    # a long-horizon difference is luck, so the report needs the spread
    members = {}
    for seed in seeds:
        _, _, pert = run(seed)
        print(f"twin: member seed={seed} done", flush=True)
        members[seed] = {
            f: np.asarray(
                [float(np.nanmax(np.abs(a[f].astype(np.float64) - b[f].astype(np.float64)))) for a, b in zip(base, pert)]
            )
            for f in prog
        }

    out = dict(
        variant=np.asarray(name),
        perturb=np.asarray(perturb),
        seeds=np.asarray(list(seeds)),
        timesteps=np.asarray(timesteps),
    )
    for f in prog:
        # shape (n_seeds, n_records)
        out[f"err_{f}_max_abs"] = np.stack([members[seed][f] for seed in seeds])
        out[f"scale_{f}"] = np.asarray(_scale(base[-1][f]))

    print(f"\n=== {name}: veros vs veros(temp * (1 + {perturb:g} * noise)), {len(seeds)} seed(s) ===")
    print("step " + "".join(f"{('seed ' + str(s)):>12s}" for s in seeds) + "   (temp)")
    for i, t in enumerate(timesteps):
        print(f"{t:6d} " + "".join(f"{out['err_temp_max_abs'][j][i]:12.3e}" for j in range(len(seeds))), flush=True)

    return out


def experiment_long(variant_name, veros_path, n_steps, record_interval, atol):
    """mini vs veros over the full matrix horizon, with the solver's stopping rule forced to `atol`."""
    from variant_util import build_mini_variant, build_real_variant

    variant = VARIANTS_BY_NAME[variant_name]
    name, family, overrides = variant["name"], variant["family"], variant["overrides"]

    with forced_solver_atol(atol):
        _, _, _, _, timesteps, mini_states = build_mini_variant(
            name, family, overrides, n_steps, veros_path, record_interval
        )
        print("long: mini done", flush=True)
        _, _, _, _, _, real_states = build_real_variant(
            name, family, overrides, n_steps, veros_path, record_interval
        )
        print("long: real done", flush=True)

    fields = sorted(mini_states[0])
    out = dict(variant=np.asarray(name), atol=np.asarray(atol), timesteps=np.asarray(timesteps))
    for f in fields:
        out[f"err_{f}_max_abs"] = np.asarray(
            [
                float(np.nanmax(np.abs(np.asarray(m[f], np.float64) - np.asarray(r[f], np.float64))))
                for m, r in zip(mini_states, real_states)
            ]
        )
        # internal variability of the reference run over the second half, the
        # scale any long-horizon difference has to be read against
        half = np.stack([np.asarray(r[f], np.float64) for r in real_states[len(real_states) // 2 :]])
        out[f"internal_std_{f}"] = np.asarray(float(np.nanmax(half.std(axis=0))))

    print(f"\n=== {name}: mini vs veros, solver atol={atol:g} ===")
    print("step " + "".join(f"{f:>12s}" for f in fields))
    for i, t in enumerate(timesteps):
        print(f"{t:6d} " + "".join(f"{out[f'err_{f}_max_abs'][i]:12.3e}" for f in fields), flush=True)

    return out


def _solver_worker(variant_name, veros_path, n_steps, record_interval, solver, out_path):
    """
    One veros run with `solver` as the linear-solver backend, saved to out_path.

    Runs in its own process (see experiment_solver): veros locks
    runtime_settings once anything imports veros.core, so two backends cannot
    coexist in one interpreter.
    """
    import importlib

    sys.path.insert(0, str(veros_path))
    from veros import runtime_settings as rs

    rs.backend = "jax"
    rs.device = "cpu"
    rs.float_type = "float64"
    rs.linear_solver = solver

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    from veros.routines import veros_routine

    from variant_util import _prognostic_fields

    variant = VARIANTS_BY_NAME[variant_name]
    spec = FAMILIES[variant["family"]]
    RealSetupClass = getattr(importlib.import_module(spec["real_module"]), spec["real_class"])

    class NoIOSetup(RealSetupClass):
        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

    sim = NoIOSetup(override=dict(runlen=0, **variant["overrides"]))
    sim.setup()
    prog = _prognostic_fields(sim.state.settings.enable_eke, sim.state.settings.enable_tke)

    def snapshot():
        vs = sim.state.variables
        return {n: np.asarray(getattr(vs, n)[..., vs.tau]) for n in prog}

    timesteps, frames = [0], [snapshot()]
    for i in range(n_steps):
        sim.step(sim.state)
        if (i + 1) % record_interval == 0:
            timesteps.append(i + 1)
            frames.append(snapshot())

    out = dict(timesteps=np.asarray(timesteps), solver=np.asarray(solver), variant=np.asarray(variant_name))
    for f in prog:
        out[f] = np.stack([fr[f] for fr in frames])
    np.savez(Path(out_path), **out)


def experiment_solver(variant_name, veros_path, n_steps, record_interval, solvers=("scipy_jax", "scipy")):
    """Real veros vs real veros, differing only in linear-solver backend."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        procs = {}
        for solver in solvers:
            paths[solver] = Path(tmp) / f"{solver}.npz"
            procs[solver] = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "solver",
                 "--variant", variant_name, "--veros-path", str(veros_path),
                 "--long-steps", str(n_steps), "--long-record-interval", str(record_interval),
                 "--_solver-worker", solver, "--_solver-out", str(paths[solver])]
            )
        for solver, proc in procs.items():
            if proc.wait() != 0:
                raise RuntimeError(f"{solver} worker failed with exit code {proc.returncode}")

        loaded = {s: dict(np.load(paths[s], allow_pickle=True)) for s in solvers}

    a, b = (loaded[s] for s in solvers)
    fields = [f for f in a if f not in ("timesteps", "solver", "variant")]
    out = dict(
        timesteps=a["timesteps"],
        variant=np.asarray(variant_name),
        solvers=np.asarray(list(solvers)),
    )
    print(f"\n=== {variant_name}: veros({solvers[0]}) vs veros({solvers[1]}) ===")
    print("step " + "".join(f"{f:>12s}" for f in sorted(fields)))
    for f in sorted(fields):
        out[f"err_{f}_max_abs"] = np.asarray(
            [float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) for x, y in zip(a[f], b[f])]
        )
    for i, t in enumerate(a["timesteps"]):
        print(f"{t:6d} " + "".join(f"{out[f'err_{f}_max_abs'][i]:12.3e}" for f in sorted(fields)), flush=True)

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=("init", "physics", "seed", "twin", "long", "solver", "all"))
    parser.add_argument("--variant", default="acc_basic", help="variant for seed/twin")
    parser.add_argument("--group", choices=("acc", "global"), default=None, help="restrict `init` to one group")
    parser.add_argument("--veros-path", type=Path, default=DEFAULT_VEROS_PATH)
    parser.add_argument("--seed-steps", type=int, default=10)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--twin-steps", type=int, default=365 * 30)
    parser.add_argument("--twin-record-interval", type=int, default=150)
    parser.add_argument("--perturb", type=float, default=1e-12)
    parser.add_argument("--twin-seeds", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="one perturbed member per seed, all against a single unperturbed baseline")
    parser.add_argument("--long-steps", type=int, default=365 * 30)
    parser.add_argument("--long-record-interval", type=int, default=150)
    parser.add_argument("--long-atol", type=float, default=TIGHT_SOLVER_ATOL)
    # internal: experiment_solver re-invokes this script per backend, since
    # veros locks its runtime settings once per interpreter
    parser.add_argument("--_solver-worker", dest="solver_worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_solver-out", dest="solver_out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.solver_worker:
        sys.path.insert(0, str(REPO_ROOT / "mini-veros"))
        _solver_worker(
            args.variant, args.veros_path, args.long_steps, args.long_record_interval,
            args.solver_worker, args.solver_out,
        )
        return

    if not (args.veros_path / "veros" / "__init__.py").exists():
        parser.error(f"no veros package found at {args.veros_path}")

    configure_veros_runtime(args.veros_path)
    sys.path.insert(0, str(REPO_ROOT / "mini-veros"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wanted = ("init", "physics", "seed", "twin", "long", "solver") if args.experiment == "all" else (args.experiment,)

    if "init" in wanted:
        out = experiment_init(args.veros_path, args.group)
        path = OUT_DIR / f"init_parity{'_' + args.group if args.group else ''}.npz"
        np.savez(path, **out)
        print(f"saved {path}")

    if "physics" in wanted:
        out = experiment_physics(args.veros_path, args.physics_steps, TIGHT_SOLVER_ATOL, args.group)
        path = OUT_DIR / f"physics_parity{'_' + args.group if args.group else ''}.npz"
        np.savez(path, **out)
        print(f"saved {path}")

    if "seed" in wanted:
        out = experiment_seed(
            args.variant, args.veros_path, args.seed_steps, (DEFAULT_SOLVER_ATOL, TIGHT_SOLVER_ATOL)
        )
        path = OUT_DIR / f"seed_{args.variant}.npz"
        np.savez(path, **out)
        print(f"saved {path}")

    if "twin" in wanted:
        out = experiment_twin(
            args.variant, args.veros_path, args.twin_steps, args.twin_record_interval, args.perturb,
            seeds=tuple(args.twin_seeds),
        )
        path = OUT_DIR / f"twin_{args.variant}_p{args.perturb:g}.npz"
        np.savez(path, **out)
        print(f"saved {path}")

    if "long" in wanted:
        out = experiment_long(
            args.variant, args.veros_path, args.long_steps, args.long_record_interval, args.long_atol
        )
        path = OUT_DIR / f"long_{args.variant}_atol{args.long_atol:g}.npz"
        np.savez(path, **out)
        print(f"saved {path}")

    if "solver" in wanted:
        out = experiment_solver(args.variant, args.veros_path, args.long_steps, args.long_record_interval)
        path = OUT_DIR / f"solver_{args.variant}.npz"
        np.savez(path, **out)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
