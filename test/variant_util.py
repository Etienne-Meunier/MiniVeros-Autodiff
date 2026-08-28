#!/usr/bin/env python3
"""
Generalized build/run for the setup matrix in setups_matrix.py: takes a
family (base geometry + baseline physics) plus a dict of settings
overrides, and builds/runs both mini_veros and real veros from the *same*
overrides dict -- see setups_matrix.py's docstring for why one dict can
drive both sides.

Mirrors util.py's build_mini/build_real contract (same return shapes), so
downstream code (compare_field, compute_error_evolution, etc. in util.py)
works unchanged on this module's output. Kept separate from util.py rather
than folded in, since util.py's build_mini/build_real are keyed by a fixed
setup-name -> real-class map (3 setups) and several other test/ scripts
still import that exact signature.
"""

import dataclasses
import importlib
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from setups_matrix import FAMILIES

PROGNOSTIC_FIELDS = ["u", "v", "temp", "salt", "psi"]


def _route_overrides(model, overrides):
    """Split `overrides` between model.config (StaticConfig) and model.parameters (Parameters) by field name."""
    config_fields = {f.name for f in dataclasses.fields(model.config)}
    param_fields = {f.name for f in dataclasses.fields(model.parameters)}
    unknown = set(overrides) - config_fields - param_fields
    if unknown:
        raise ValueError(f"unknown override keys (not a StaticConfig or Parameters field): {sorted(unknown)}")

    config_over = {k: v for k, v in overrides.items() if k in config_fields}
    param_over = {k: v for k, v in overrides.items() if k in param_fields}
    new_config = dataclasses.replace(model.config, **config_over) if config_over else model.config
    new_params = dataclasses.replace(model.parameters, **param_over) if param_over else model.parameters
    return dataclasses.replace(model, config=new_config, parameters=new_params)


def _prognostic_fields(enable_eke, enable_tke):
    # real veros raises "Variable X is not active in this configuration" on
    # any field whose enable_* flag is off, rather than returning zeros --
    # so both eke and tke have to be conditional here, not just eke.
    return PROGNOSTIC_FIELDS + (["tke"] if enable_tke else []) + (["eke"] if enable_eke else [])


def _block(tree):
    """Force every array leaf to materialize, so wall-clock timing reflects actual device compute, not async dispatch."""
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return tree


def _run_steps(model, step0, forcing_fn, prog_fields, n_steps, record_interval, time_n_steps):
    """
    Hand-rolled per-step jax.jit loop.

    loop.run's NaN-check short circuit (lax.cond between "already stopped"
    and "do one step") requires both branches to return an IntegratorState
    with identical pytree structure -- but Tendencies/DiagnosticState fields
    default to None and are only populated by whichever physics terms are
    active (e.g. dtke, dtemp_mix, dpsi, K_diss_h), so the *initial*
    integrator_state's None-pattern only matches "after one real step"'s
    None-pattern for the exact baseline configs the setups were written
    against (acc_basic, acc_full, global_4deg's defaults) -- any override
    that changes which terms are active/inactive breaks that match and
    lax.cond raises "branch outputs must have the same pytree structure".
    A plain per-step jax.jit call has no such branch, so it works
    uniformly across the whole matrix; loop.run's single-compiled-scan is
    only safe for those three unmodified setups (see their own __main__
    blocks and generate_pressure_report_data.py, which already uses this
    same per-step pattern for exactly this reason on the pressure-solver
    variant).
    """
    import jax

    from mini_veros import loop

    step_jit = jax.jit(lambda s: loop.step(model, s, forcing_fn(model, s.state)))
    _block(step_jit(step0))  # trace/compile, discarded -- n_steps below must start from step0, not this warm-up step
    step = step0

    timesteps, recorded_states = [], []
    if record_interval is not None:
        timesteps.append(0)
        recorded_states.append({name_: np.asarray(getattr(step0.state, name_)) for name_ in prog_fields})

    for i in range(n_steps):
        step = step_jit(step)
        if record_interval is not None and (i + 1) % record_interval == 0:
            step = _block(step)
            timesteps.append(i + 1)
            recorded_states.append({name_: np.asarray(getattr(step.state, name_)) for name_ in prog_fields})

    step = _block(step)
    state_final = {name_: np.asarray(getattr(step.state, name_)) for name_ in prog_fields}

    sec_per_step = None
    if time_n_steps > 0:
        t0 = time.perf_counter()
        for _ in range(time_n_steps):
            step = step_jit(step)
        _block(step)
        sec_per_step = (time.perf_counter() - t0) / time_n_steps

    return state_final, sec_per_step, timesteps, recorded_states


def build_mini_variant(name, family, overrides, n_steps, veros_path, record_interval=None, time_n_steps=0):
    """
    Build mini_veros for `family`, apply `overrides`, and run n_steps.

    Returns (model, state_initial, state_final, sec_per_step, timesteps, recorded_states).
    timesteps/recorded_states are [] unless record_interval is set.
    sec_per_step is None unless time_n_steps > 0.
    """
    sys.path.insert(0, str(REPO_ROOT / "mini-veros"))
    sys.path.insert(0, str(veros_path))

    import jax
    jax.config.update("jax_enable_x64", True)

    from mini_veros.state import Tendencies

    setup_mod = importlib.import_module(FAMILIES[family]["mini_module"])
    model, step0, forcing_fn = setup_mod.build()
    model = _route_overrides(model, overrides)

    # step0.tendency_m1/m2 were zero-inited against the pre-override config
    # by setup_mod.build() -- Tendencies.init's field population is gated on
    # config flags (see state.py), so an override that flips a gate (e.g.
    # enable_tke_superbee_advection) leaves fields the new config expects
    # (e.g. dtke) as None instead of zero. Re-init against the final config;
    # safe since these are all-zero AB2 seed values at t=0 either way.
    nisle = model.boundary_conditions.psin.shape[-1]
    zero_tend = Tendencies.init(model.config, nisle)
    step0 = dataclasses.replace(step0, tendency_m1=zero_tend, tendency_m2=zero_tend)

    prog_fields = _prognostic_fields(model.config.enable_eke, model.config.enable_tke)
    state_initial = {name_: np.asarray(getattr(step0.state, name_)) for name_ in prog_fields}

    state_final, sec_per_step, timesteps, recorded_states = _run_steps(
        model, step0, forcing_fn, prog_fields, n_steps, record_interval, time_n_steps
    )

    return model, state_initial, state_final, sec_per_step, timesteps, recorded_states


def build_real_variant(name, family, overrides, n_steps, veros_path, record_interval=None, time_n_steps=0):
    """
    Build real veros for `family`, apply `overrides` via VerosSetup's
    `override=` dict, and run n_steps. Same return contract as
    build_mini_variant, minus the `model` (returns `sim` instead).
    """
    sys.path.insert(0, str(veros_path))

    from veros.routines import veros_routine

    spec = FAMILIES[family]
    real_mod = importlib.import_module(spec["real_module"])
    RealSetupClass = getattr(real_mod, spec["real_class"])

    class NoIOSetup(RealSetupClass):
        # never touch disk regardless of steps -- set_diagnostics only
        # configures *what* to write; clearing it disables all output/restart
        # writes (same pattern veros's own pyom_consistency/acc_test.py uses)
        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

    sim = NoIOSetup(override=dict(runlen=0, **overrides))
    sim.setup()

    prog_fields = _prognostic_fields(sim.state.settings.enable_eke, sim.state.settings.enable_tke)

    def field_now():
        vs = sim.state.variables
        return {name_: np.asarray(getattr(vs, name_)[..., vs.tau]) for name_ in prog_fields}

    state_initial = field_now()

    timesteps, recorded_states = [], []
    if record_interval is not None:
        timesteps.append(0)
        recorded_states.append(state_initial.copy())

    for i in range(n_steps):
        sim.step(sim.state)
        if record_interval is not None and (i + 1) % record_interval == 0:
            timesteps.append(i + 1)
            recorded_states.append(field_now())

    state_final = field_now()

    sec_per_step = None
    if time_n_steps > 0:
        t0 = time.perf_counter()
        for _ in range(time_n_steps):
            sim.step(sim.state)
        np.asarray(sim.state.variables.temp)  # force sync before stopping the clock
        sec_per_step = (time.perf_counter() - t0) / time_n_steps

    return sim, state_initial, state_final, sec_per_step, timesteps, recorded_states
