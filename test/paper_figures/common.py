"""
Shared pieces for the paper experiments.

Every experiment runs the same model: global_4deg, a realistic global ocean at
4-degree resolution (90x40x15) with real bathymetry and climatological
forcing, dt_tracer = 86400 s, so one step is one day and 365 steps is a model
year. All of it starts from the same 30-model-year spin-up, cached to disk so
it is computed once.

The layout is always the same. `expN_*.py` runs the simulations and writes one
.npz into DATA_DIR; `figN_*.py` reads that file back and writes a pdf into
FIG_DIR. Nothing in mini-veros is modified: the experiments only ever replace
entries of `model.parameters` / `model.config`, or fields of the initial
state, or wrap `forcing_fn`.
"""

import dataclasses
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)

from mini_veros import loop
from mini_veros.setups.global_4deg.default import build

REPO = Path(__file__).resolve().parents[2]
STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
DATA_DIR = STORE / "MiniVeros-Autodiff" / "results" / "paper"
FIG_DIR = REPO / "report" / "paper" / "figures"

SPINUP_YEARS = 30
SPINUP_STEPS = SPINUP_YEARS * 365  # dt_tracer = 1 day
SPINUP_CACHE = DATA_DIR / f"spinup_global4deg_{SPINUP_YEARS}y.eqx"

# exp1_gradient_check's horizon: the longest rollout (in days) where autodiff
# still agrees with finite differences to within 1%, for every checked
# parameter. Filled in by hand once exp1 has run; every later experiment sizes
# its rollout off this constant rather than a re-guessed literal.
# Measured on global_4deg, 30-year spin-up: matches to 5e-3 at 80 days, then
# 5.6x too large at 160 days (exp1_gradient_check.npz).
HORIZON_DAYS = 80


def spinup(n_steps: int = SPINUP_STEPS, overrides: dict | None = None):
    """Build global_4deg (optionally with config/parameter overrides, see model.apply_overrides)
    and integrate it forward, so experiments start from a settled state.

    Cached to disk -- SPINUP_CACHE for the default config, one file per distinct
    `overrides` otherwise -- so the first call computes and saves it and every
    later call (this process or a new one) loads it back in seconds. Returns
    (model, integrator_state, forcing_fn), the same triple `build()` returns.
    """
    cache = SPINUP_CACHE if not overrides else DATA_DIR / (
        f"spinup_global4deg_{SPINUP_YEARS}y__" + "_".join(f"{k}={v}" for k, v in sorted(overrides.items())) + ".eqx"
    )
    model, integrator_state, forcing_fn = build(overrides)
    if cache.exists():
        integrator_state = eqx.tree_deserialise_leaves(cache, integrator_state)
        print(f"spinup: loaded {cache}")
        return model, integrator_state, forcing_fn

    integrator_state, _ = eqx.filter_jit(loop.run)(
        model, integrator_state, forcing_fn, lambda s: None, n_steps, n_steps
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(cache, integrator_state)
    print(f"spinup: ran {n_steps} steps, saved {cache}")
    return model, integrator_state, forcing_fn


def with_params(model, params: dict):
    """Copy of `model` with the named scalar parameters replaced (e.g. {"c_k": 0.12})."""
    return dataclasses.replace(model, parameters=dataclasses.replace(model.parameters, **params))


def safe_checkpoint_every(n_steps: int, target: int = 10) -> int:
    """The largest divisor of n_steps that is <= target -- a `checkpoint_every` that always works,
    for callers whose n_steps is derived from HORIZON_DAYS and so isn't guaranteed to be a
    round number.
    """
    for c in range(min(target, n_steps), 0, -1):
        if n_steps % c == 0:
            return c
    return 1


def rollout(model, integrator_state, forcing_fn, n_steps: int, checkpoint_every: int | None = None):
    """Integrate n_steps and return the final IntegratorState.

    Blocks of `checkpoint_every` steps are rematerialised rather than stored, which is
    what makes reverse-mode differentiation over thousands of steps fit in memory.
    Peak memory is one state per block; the price is one extra forward pass.
    `checkpoint_every` defaults to `safe_checkpoint_every(n_steps)` (up to 10).
    """
    if checkpoint_every is None:
        checkpoint_every = safe_checkpoint_every(n_steps)
    assert n_steps % checkpoint_every == 0, "n_steps must be a multiple of checkpoint_every"

    def one_step(integrator_state, _):
        force = forcing_fn(model, integrator_state.state)
        return loop.step(model, integrator_state, force), None

    @jax.checkpoint
    def block(integrator_state, _):
        integrator_state, _ = lax.scan(one_step, integrator_state, length=checkpoint_every)
        return integrator_state, None

    integrator_state, _ = lax.scan(block, integrator_state, length=n_steps // checkpoint_every)
    return integrator_state


def interior(field):
    """Drop the two ghost cells on each horizontal edge."""
    return field[2:-2, 2:-2]


def sst(state):
    """Sea surface temperature on the interior grid (the model's top level is index -1)."""
    return interior(state.temp)[..., -1]


def mean_square_sst(state):
    """The scalar loss used to check gradients: mean of squared surface temperature."""
    return (sst(state) ** 2).mean()


def upper_ts(state, n_layers: int = 3):
    """Interior (temp, salt) of the top `n_layers` levels (top level is index -1)."""
    return interior(state.temp)[..., -n_layers:], interior(state.salt)[..., -n_layers:]


def upper_ts_scale(model, state, n_layers: int = 3):
    """Per-(variable, layer) normaliser for `upper_ts_misfit`: the ocean variance of `state`'s
    own top `n_layers` levels of temp/salt, layer by layer. Computed once from the reference
    (e.g. spin-up or true) state and reused for every misfit evaluation against it.
    """
    ocean = interior(model.boundary_conditions.maskT)[..., -n_layers:] != 0
    temp, salt = upper_ts(state, n_layers)
    n_ocean = jnp.maximum(ocean.sum(axis=(0, 1)), 1)

    def var(field):
        mean = jnp.where(ocean, field, 0.0).sum(axis=(0, 1)) / n_ocean
        return jnp.where(ocean, (field - mean) ** 2, 0.0).sum(axis=(0, 1)) / n_ocean

    return var(temp), var(salt)


def upper_ts_misfit(model, state, target_temp, target_salt, scale):
    """Sum over variable and layer of mean_ocean((X - X_target)^2) / scale[var, layer].

    `target_temp`/`target_salt` are the top-`n_layers` fields of the reference state
    (e.g. from `upper_ts`), `scale` is `(scale_temp, scale_salt)` from `upper_ts_scale`
    evaluated on that same reference -- this makes temperature and salinity
    dimensionless and comparable, since c_k/c_eps act on stratification (the density
    contrast between the top layers) rather than on either field alone.
    """
    n_layers = target_temp.shape[-1]
    ocean = interior(model.boundary_conditions.maskT)[..., -n_layers:] != 0
    n_ocean = jnp.maximum(ocean.sum(axis=(0, 1)), 1)
    temp, salt = upper_ts(state, n_layers)
    scale_temp, scale_salt = scale

    def mean_sq_err(field, target, s):
        err2 = jnp.where(ocean, (field - target) ** 2, 0.0).sum(axis=(0, 1)) / n_ocean
        return (err2 / s).sum()

    return mean_sq_err(temp, target_temp, scale_temp) + mean_sq_err(salt, target_salt, scale_salt)


def save(name: str, **arrays):
    """Write one experiment's output to DATA_DIR/<name>.npz."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.npz"
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print(f"wrote {path}")


def load(name: str):
    """Read back what `save` wrote."""
    path = DATA_DIR / f"{name}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the matching exp*.py first")
    return np.load(path)


def grid_arrays(model):
    """Interior longitudes, latitudes and surface land mask, for the map figures."""
    x = np.asarray(model.grid.xt[2:-2])
    y = np.asarray(model.grid.yt[2:-2])
    land = np.asarray(interior(model.boundary_conditions.maskT)[..., -1]) == 0
    return x, y, land
