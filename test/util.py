#!/usr/bin/env python3
"""
Utilities for comparing mini_veros and veros implementations.

This module provides simple, composable functions to:
  - Set up mini_veros and real veros with the same configuration
  - Run both models for N steps
  - Extract and compare prognostic state variables
"""

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

# Prognostic fields that both implementations evolve
PROGNOSTIC_FIELDS = ["u", "v", "temp", "salt", "psi", "tke"]
EKE_SETUPS = ("acc", "global_4deg")


def configure_veros_runtime(veros_path):
    """
    Configure real veros runtime settings before any veros.core imports.

    Must run exactly once, before importing veros.core or any setup that
    imports veros.tools (which transitively imports veros.core).

    Args:
        veros_path: Path to the real veros repository.
    """
    sys.path.insert(0, str(veros_path))

    from veros import runtime_settings as rs

    rs.backend = "jax"
    rs.device = "cpu"
    rs.float_type = "float64"
    rs.linear_solver = "scipy_jax"

    import jax
    jax.config.update("jax_enable_x64", True)


def build_mini(setup_name, n_steps, veros_path, record_interval=None):
    """
    Build and run mini_veros for n_steps.

    Args:
        setup_name: Setup name (e.g., "acc_basic", "acc").
        n_steps: Number of steps to run.
        veros_path: Path to real veros repo (needed for global_4deg imports).
        record_interval: If not None, record state every N steps (including step 0).

    Returns:
        (model, state_initial, state_final): model object, initial state dict,
        and final state dict after n_steps. State dicts contain only prognostic
        fields (u, v, temp, salt, psi, tke, and eke if applicable).

        If record_interval is set, returns:
        (model, state_initial, state_final, timesteps, recorded_states) where:
          - timesteps: list of step indices where states were recorded
          - recorded_states: list of state dicts at those timesteps
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(veros_path))

    import jax
    jax.config.update("jax_enable_x64", True)

    from mini_veros import loop
    import importlib

    setup_mod = importlib.import_module(_MINI_SETUP_MODULE.get(setup_name, f"mini_veros.setups.{setup_name}"))
    model, step0, forcing_fn = setup_mod.build()

    prog_fields = _get_prognostic_fields(setup_name)

    # Initial state
    state_initial = {
        name: np.asarray(getattr(step0.state, name)) for name in prog_fields
    }

    # JIT compile the step function
    import jax
    step_jit = jax.jit(lambda s: loop.step(model, s, forcing_fn))
    step_jit(step0)

    # Run steps
    step = step0
    timesteps = []
    recorded_states = []

    if record_interval is not None:
        timesteps.append(0)
        recorded_states.append(state_initial.copy())

    for i in range(n_steps):
        step = step_jit(step)
        if record_interval is not None and (i + 1) % record_interval == 0:
            timesteps.append(i + 1)
            state_at_i = {
                name: np.asarray(getattr(step.state, name)) for name in prog_fields
            }
            recorded_states.append(state_at_i)

    # Final state
    state_final = {
        name: np.asarray(getattr(step.state, name)) for name in prog_fields
    }

    if record_interval is not None:
        return model, state_initial, state_final, timesteps, recorded_states
    return model, state_initial, state_final


def build_real(setup_name, n_steps, veros_path, setup_class_map, record_interval=None):
    """
    Build and run real veros for n_steps.

    Args:
        setup_name: Setup name (e.g., "acc_basic", "acc").
        n_steps: Number of steps to run.
        veros_path: Path to the real veros repository.
        setup_class_map: Dict mapping setup_name to real veros class name.
        record_interval: If not None, record state every N steps (including step 0).

    Returns:
        (sim, state_initial, state_final): sim object, initial state dict,
        and final state dict after n_steps. State dicts contain only prognostic
        fields (u, v, temp, salt, psi, tke, and eke if applicable).

        If record_interval is set, returns:
        (sim, state_initial, state_final, timesteps, recorded_states) where:
          - timesteps: list of step indices where states were recorded
          - recorded_states: list of state dicts at those timesteps
    """
    sys.path.insert(0, str(veros_path))

    from veros.routines import veros_routine
    import importlib

    real_setup_mod = importlib.import_module(
        f"veros.setups.{setup_name}.{setup_name}"
    )
    RealSetupClass = getattr(real_setup_mod, setup_class_map[setup_name])

    class NoIOSetup(RealSetupClass):
        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

    sim = NoIOSetup(override=dict(runlen=0))
    sim.setup()

    vs = sim.state.variables
    prog_fields = _get_prognostic_fields(setup_name)

    # Initial state
    state_initial = {
        name: np.asarray(getattr(vs, name)[..., vs.tau]) for name in prog_fields
    }

    # Run steps
    timesteps = []
    recorded_states = []

    if record_interval is not None:
        timesteps.append(0)
        recorded_states.append(state_initial.copy())

    for i in range(n_steps):
        sim.step(sim.state)
        if record_interval is not None and (i + 1) % record_interval == 0:
            timesteps.append(i + 1)
            vs = sim.state.variables
            state_at_i = {
                name: np.asarray(getattr(vs, name)[..., vs.tau]) for name in prog_fields
            }
            recorded_states.append(state_at_i)

    # Final state
    vs = sim.state.variables
    state_final = {
        name: np.asarray(getattr(vs, name)[..., vs.tau]) for name in prog_fields
    }

    if record_interval is not None:
        return sim, state_initial, state_final, timesteps, recorded_states
    return sim, state_initial, state_final


def compare_field(name, mini_array, real_array, atol=1e-8, rtol=1e-8):
    """
    Compare a single prognostic field between mini_veros and real veros.

    Args:
        name: Field name (e.g., "u", "temp").
        mini_array: numpy array from mini_veros.
        real_array: numpy array from real veros.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        dict with keys: name, ok (bool), shape, max_abs, max_rel, argmax,
        mini_at_argmax, real_at_argmax. If shapes don't match, includes
        an 'error' key instead of the comparison metrics.
    """
    if mini_array.shape != real_array.shape:
        return dict(
            name=name,
            ok=False,
            error=f"shape mismatch: mini={mini_array.shape} real={real_array.shape}",
        )

    mini_array = mini_array.astype(np.float64)
    real_array = real_array.astype(np.float64)
    diff = np.abs(mini_array - real_array)
    denom = np.maximum(np.abs(mini_array), np.abs(real_array))
    denom = np.where(denom == 0, 1.0, denom)
    rel = diff / denom

    ok = bool(np.allclose(mini_array, real_array, atol=atol, rtol=rtol, equal_nan=True))
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0
    max_rel = float(np.nanmax(rel)) if diff.size else 0.0
    mean_abs = float(np.nanmean(diff)) if diff.size else 0.0
    median_abs = float(np.nanmedian(diff)) if diff.size else 0.0
    mean_rel = float(np.nanmean(rel)) if diff.size else 0.0
    median_rel = float(np.nanmedian(rel)) if diff.size else 0.0
    argmax = (
        np.unravel_index(np.nanargmax(diff), diff.shape) if diff.size else None
    )

    return dict(
        name=name,
        ok=ok,
        shape=mini_array.shape,
        max_abs=max_abs,
        max_rel=max_rel,
        mean_abs=mean_abs,
        median_abs=median_abs,
        mean_rel=mean_rel,
        median_rel=median_rel,
        argmax=argmax,
        mini_at_argmax=float(mini_array[argmax]) if argmax is not None else None,
        real_at_argmax=float(real_array[argmax]) if argmax is not None else None,
    )


def compute_error_evolution(timesteps, mini_states, real_states, atol=1e-8, rtol=1e-8):
    """
    Compute errors for each field at each timestep.

    Args:
        timesteps: List of step indices.
        mini_states: List of mini_veros state dicts (one per timestep).
        real_states: List of real veros state dicts (one per timestep).
        atol: Absolute tolerance for pass/fail determination.
        rtol: Relative tolerance for pass/fail determination.

    Returns:
        dict mapping field_name -> {
            "timesteps": list of timesteps,
            "max_abs_errors": list of max abs errors at each timestep,
            "max_rel_errors": list of max rel errors at each timestep,
            "mean_abs_errors": list of mean abs errors at each timestep,
            "median_abs_errors": list of median abs errors at each timestep,
            "mean_rel_errors": list of mean rel errors at each timestep,
            "median_rel_errors": list of median rel errors at each timestep,
            "passes": list of bools indicating pass at each timestep
        }
    """
    fields_errors = {}

    # Collect all field names
    all_fields = set()
    for mini_state in mini_states:
        all_fields.update(mini_state.keys())

    for field_name in sorted(all_fields):
        max_abs_errs = []
        max_rel_errs = []
        mean_abs_errs = []
        median_abs_errs = []
        mean_rel_errs = []
        median_rel_errs = []
        passes = []

        for mini_state, real_state in zip(mini_states, real_states):
            if field_name not in mini_state or field_name not in real_state:
                max_abs_errs.append(0.0)
                max_rel_errs.append(0.0)
                mean_abs_errs.append(0.0)
                median_abs_errs.append(0.0)
                mean_rel_errs.append(0.0)
                median_rel_errs.append(0.0)
                passes.append(True)
                continue

            result = compare_field(
                field_name,
                mini_state[field_name],
                real_state[field_name],
                atol,
                rtol,
            )
            if "error" in result:
                max_abs_errs.append(0.0)
                max_rel_errs.append(0.0)
                mean_abs_errs.append(0.0)
                median_abs_errs.append(0.0)
                mean_rel_errs.append(0.0)
                median_rel_errs.append(0.0)
                passes.append(False)
            else:
                max_abs_errs.append(result["max_abs"])
                max_rel_errs.append(result["max_rel"])
                mean_abs_errs.append(result["mean_abs"])
                median_abs_errs.append(result["median_abs"])
                mean_rel_errs.append(result["mean_rel"])
                median_rel_errs.append(result["median_rel"])
                passes.append(result["ok"])

        fields_errors[field_name] = {
            "timesteps": timesteps,
            "max_abs_errors": max_abs_errs,
            "max_rel_errors": max_rel_errs,
            "mean_abs_errors": mean_abs_errs,
            "median_abs_errors": median_abs_errs,
            "mean_rel_errors": mean_rel_errs,
            "median_rel_errors": median_rel_errs,
            "passes": passes,
        }

    return fields_errors


def print_error_evolution(fields_errors, rtol=1e-8):
    """
    Print error evolution table for each field.

    Args:
        fields_errors: Dict returned by compute_error_evolution.
        rtol: Relative tolerance for highlighting.
    """
    print("\n=== Error Evolution Over Time ===\n")

    for field_name in sorted(fields_errors.keys()):
        data = fields_errors[field_name]
        timesteps = data["timesteps"]
        max_rel_errors = data["max_rel_errors"]
        mean_rel_errors = data["mean_rel_errors"]
        median_rel_errors = data["median_rel_errors"]
        passes = data["passes"]

        print(f"{field_name}:")
        print(f"  Timestep | Max Rel Error | Mean Rel Error | Median Rel Error | Pass")
        print(f"  " + "-" * 72)
        for t, err, mean_err, median_err, p in zip(
            timesteps, max_rel_errors, mean_rel_errors, median_rel_errors, passes
        ):
            status = "✓" if p else "✗"
            flag = " *" if err > rtol else ""
            print(
                f"  {t:8d} | {err:13.3e} | {mean_err:14.3e} | {median_err:16.3e} | "
                f"{status}{flag}"
            )
        print()

    # Summary across all fields, per timestep
    field_names = sorted(fields_errors.keys())
    if field_names:
        timesteps = fields_errors[field_names[0]]["timesteps"]
        print("ALL FIELDS (avg / median across fields):")
        print(
            f"  Timestep | Avg Max Rel | Med Max Rel | Avg Mean Rel | Med Mean Rel "
            f"| Avg Median Rel | Med Median Rel"
        )
        print(f"  " + "-" * 96)
        for i, t in enumerate(timesteps):
            max_rel_vals = [fields_errors[f]["max_rel_errors"][i] for f in field_names]
            mean_rel_vals = [fields_errors[f]["mean_rel_errors"][i] for f in field_names]
            median_rel_vals = [
                fields_errors[f]["median_rel_errors"][i] for f in field_names
            ]
            print(
                f"  {t:8d} | {np.mean(max_rel_vals):11.3e} | {np.median(max_rel_vals):11.3e} | "
                f"{np.mean(mean_rel_vals):12.3e} | {np.median(mean_rel_vals):12.3e} | "
                f"{np.mean(median_rel_vals):14.3e} | {np.median(median_rel_vals):14.3e}"
            )
        print()


def _get_prognostic_fields(setup_name):
    """Return list of prognostic fields for a given setup."""
    fields = list(PROGNOSTIC_FIELDS)
    if setup_name in EKE_SETUPS:
        fields.append("eke")
    return fields


def print_section(title, mini_dict, real_dict, atol, rtol):
    """
    Print comparison results for a set of fields.

    Args:
        title: Section title to print.
        mini_dict: Dict of field_name -> array from mini_veros.
        real_dict: Dict of field_name -> array from real veros.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.

    Returns:
        bool: True if all fields pass, False otherwise.
    """
    print(f"\n=== {title} ===")
    all_ok = True
    mean_abs_list = []
    median_abs_list = []
    mean_rel_list = []
    median_rel_list = []
    for name in sorted(mini_dict.keys()):
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
            f"  {name:12s}  {status}  max_abs={r['max_abs']:.3e}  "
            f"max_rel={r['max_rel']:.3e}  shape={r['shape']}"
        )
        print(
            f"               mean_abs={r['mean_abs']:.3e}  median_abs={r['median_abs']:.3e}  "
            f"mean_rel={r['mean_rel']:.3e}  median_rel={r['median_rel']:.3e}"
        )
        if not r["ok"]:
            print(
                f"               worst at {r['argmax']}: mini={r['mini_at_argmax']!r} "
                f"real={r['real_at_argmax']!r}"
            )
        all_ok = all_ok and r["ok"]
        mean_abs_list.append(r["mean_abs"])
        median_abs_list.append(r["median_abs"])
        mean_rel_list.append(r["mean_rel"])
        median_rel_list.append(r["median_rel"])

    if mean_abs_list:
        # summary across all fields: avg/median of each field's own mean and median error
        print(f"  {'ALL FIELDS':12s}")
        print(
            f"               avg(mean_abs)={np.mean(mean_abs_list):.3e}  "
            f"median(mean_abs)={np.median(mean_abs_list):.3e}"
        )
        print(
            f"               avg(median_abs)={np.mean(median_abs_list):.3e}  "
            f"median(median_abs)={np.median(median_abs_list):.3e}"
        )
        print(
            f"               avg(mean_rel)={np.mean(mean_rel_list):.3e}  "
            f"median(mean_rel)={np.median(mean_rel_list):.3e}"
        )
        print(
            f"               avg(median_rel)={np.mean(median_rel_list):.3e}  "
            f"median(median_rel)={np.median(median_rel_list):.3e}"
        )

    return all_ok
