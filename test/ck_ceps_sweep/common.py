"""
Shared pieces for the c_k / c_eps density sweep: builds the acc/full ACC channel
config with the TKE closure's production/dissipation constants (c_k, c_eps)
overridden, runs it forward until the climate settles, and logs a horizontally-
averaged potential density profile (not the raw fields) every 30 days so a
30-year run stays cheap to store.

Layout mirrors test/paper_figures: run_sweep.py writes one .npz into DATA_DIR,
fig_density_profiles.py reads it back and writes report/ck_ceps_density_figures/.
Nothing in mini-veros is modified: only Parameters entries (c_k, c_eps) are
replaced via full.build's overrides dict.
"""

import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mini_veros import loop
from mini_veros.core.density.get_rho import get_potential_rho
from mini_veros.setups.acc import full

REPO = Path(__file__).resolve().parents[2]
STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
DATA_DIR = STORE / "MiniVeros-Autodiff" / "results" / "ck_ceps_sweep"
FIG_DIR = REPO / "report" / "ck_ceps_density_figures"
SWEEP_NPZ = DATA_DIR / "ck_ceps_density_sweep.npz"

# The 100-model-year, 10x10-grid full-state sweep (wandb run mnk965ig) -- superseded the
# original 30-year TOP5 full-state runs as the report figures' data source. Same .npz
# schema as run_top5_full_state.py's output (u,v,temp,salt,psi,tke,eke,zt,n_years,...),
# just more points and a longer run, stored externally (too big for $STORE/git).
EXTERNAL_100Y_DIR = Path("/Volumes/LoCe/MiniVeros-Autodiff/results/ck_ceps_100y_sweep/full_state/mnk965ig")


def full_state_path(c_k: float, c_eps: float) -> Path:
    """Path to a (c_k, c_eps) point's full-state .npz in the current data source
    (EXTERNAL_100Y_DIR). Centralizes the path so switching data source touches one line.
    """
    return EXTERNAL_100Y_DIR / f"ck{c_k:.4g}_eps{c_eps:.4g}.npz"

# dt_tracer = 43200s (12h) -> 2 steps/day, 730 steps/model-year.
STEPS_PER_DAY = int(round(86400.0 / 43200.0))
STEPS_PER_YEAR = STEPS_PER_DAY * 365
N_YEARS = 30
N_STEPS = N_YEARS * STEPS_PER_YEAR          # 21900
LOG_EVERY_DAYS = 30
LOG_EVERY = LOG_EVERY_DAYS * STEPS_PER_DAY  # 60 steps
N_LOGS = N_STEPS // LOG_EVERY               # 365
KEEP_LAST = 12                              # ~last model year, averaged for the final profile

C_K_DEFAULT = 0.10
C_EPS_DEFAULT = 0.70
# Factor-16 range around the defaults, 5 points/axis (log2-spaced), 25 runs total.
C_K_GRID = tuple(C_K_DEFAULT * 2.0**e for e in (-2, -1, 0, 1, 2))
C_EPS_GRID = tuple(C_EPS_DEFAULT * 2.0**e for e in (-2, -1, 0, 1, 2))

# Originally the 5 (c_k, c_eps) points from the 25-point sweep with the largest RMS
# deviation of their final density profile from the grid-mean profile (see
# fig_profiles_all_anomaly.png and run_top5_full_state.py's ranking snippet). The report
# figures now source from EXTERNAL_100Y_DIR's 10x10 grid instead, which doesn't contain
# these exact values, so each is mapped to its nearest neighbour on that grid (ties
# broken by always rounding up, per what the values were checked against):
#   (0.4, 0.175) -> (0.504, 0.2205)   (0.4, 0.35)  -> (0.504, 0.35)
#   (0.2, 0.175) -> (0.2, 0.2205)     (0.05, 2.8)  -> (0.05, 3.528)
#   (0.05, 1.4)  -> (0.05, 1.4)       (exact match)
TOP5 = (
    (0.504, 0.2205),
    (0.504, 0.35),
    (0.2, 0.2205),
    (0.05, 3.528),
    (0.05, 1.4),
)


def make_log_select_fn(area_t: jnp.ndarray, ocean_mask: jnp.ndarray, eq_of_state_type: int):
    """Build a log_select_fn that reduces the full 3D state to a (nz,) horizontally-averaged
    potential density profile (area-weighted, ocean cells only, referenced to the surface).

    area_t/ocean_mask are fixed for every run in the sweep (same grid, only Parameters
    change), so this closure is built once and reused across all 25 runs -- passing a fresh
    closure per run would defeat eqx.filter_jit's cache and force 25 retraces instead of one.
    """
    weight = area_t[:, :, None] * ocean_mask  # (nx+4, ny+4, 1) * (nx+4, ny+4, nz)
    denom = jnp.maximum(weight.sum(axis=(0, 1)), 1e-30)

    def log_select_fn(integrator_state):
        S = integrator_state.state
        # get_potential_rho returns a density ANOMALY (rel. rho0=1024, see
        # core/density/nonlinear_eq2.py), not absolute density.
        rho = get_potential_rho(eq_of_state_type, S.salt, S.temp, 0.0)
        return (rho * weight).sum(axis=(0, 1)) / denom

    return log_select_fn


def rolling_std(x, window=7):
    """Centred rolling std of a 1D series (edge-padded), window in samples.

    Used to shade each run's own line with its own local temporal noise --
    how much that run's series wiggles near each point in time -- as opposed
    to spatial spread (too large a range, dominates the plot) or cross-run
    spread (one shared band, not per-run colour).
    """
    x = np.asarray(x, dtype=float)
    half = window // 2
    xp = np.pad(x, half, mode="edge")
    return np.array([xp[i:i + window].std() for i in range(len(x))])


def weighted_stats(field, weight):
    """Weighted mean and std of `field` over its trailing spatial axes, per leading (time) index.

    field: (T, *spatial). weight: (*spatial,), matching field's trailing shape (e.g. a
    volume or area weight, already zeroed outside the region/mask of interest). Returns
    (mean, std), each (T,) -- used to draw a mean +/- std shaded band per run/field.
    """
    axes = tuple(range(1, field.ndim))
    denom = weight.sum()
    mean = (field * weight).sum(axis=axes) / denom
    shape = (-1,) + (1,) * (field.ndim - 1)
    var = (weight * (field - mean.reshape(shape)) ** 2).sum(axis=axes) / denom
    return mean, np.sqrt(var)


def full_state_log_select_fn(integrator_state):
    """log_select_fn that keeps the whole PrognosticState (u, v, temp, salt, tke, eke, psi)
    each logged step, instead of a reduced profile. Used by run_top5_full_state.py, which
    only runs a handful of points -- keeping the full state at every logged step for all 25
    sweep points would be ~10 GB (see the memory estimate in the report).
    """
    return integrator_state.state


def build_sweep_reductor():
    """Build the (zt, log_select_fn, run_fn) shared by every point in the sweep.

    Grid, topography, and eq_of_state_type are identical for every (c_k, c_eps) --
    only Parameters change -- so log_select_fn and the jitted run_fn are built once
    here and reused for all 25 runs. Reusing the same Python callables (rather than a
    fresh closure per run) matters for real: eqx.filter_jit caches its trace by the
    identity of non-array arguments like log_select_fn, so a fresh closure per call
    would force a full retrace on every one of the 25 runs instead of just the first.
    """
    ref_model, _, _ = full.build({})
    ocean_mask = (ref_model.boundary_conditions.maskT != 0).astype(jnp.float64)
    log_select_fn = make_log_select_fn(ref_model.grid.area_t, ocean_mask, ref_model.config.eq_of_state_type)
    run_fn = eqx.filter_jit(loop.run)
    return ref_model.grid.zt, log_select_fn, run_fn


def run_one(c_k: float, c_eps: float, log_select_fn, run_fn, n_steps: int = N_STEPS):
    """Build acc/full with (c_k, c_eps) overridden and run n_steps forward from a cold start.

    log_select_fn/run_fn come from build_sweep_reductor(), shared across the sweep so the
    model is only JIT-compiled once. Returns (profiles, diverged): profiles is
    (n_steps // LOG_EVERY, nz) of horizontally-averaged potential density, one row every
    LOG_EVERY steps; diverged is True if the run hit a non-finite state (loop.run freezes
    the state and eqx.error_if raises on return).
    """
    model, state0, forcing_fn = full.build({"c_k": c_k, "c_eps": c_eps})
    try:
        _, profiles = run_fn(model, state0, forcing_fn, log_select_fn, n_steps, LOG_EVERY)
        profiles = jnp.asarray(profiles)
        diverged = bool(jnp.any(~jnp.isfinite(profiles)))
    except eqx.EquinoxRuntimeError as e:
        print(f"  c_k={c_k:.4g} c_eps={c_eps:.4g}: stopped ({e})")
        return None, True
    return profiles, diverged
