"""
Shared pieces for the 100-model-year c_k/c_eps full-state sweep: same acc/full
channel and (c_k, c_eps) grid as test/ck_ceps_sweep/ (the 30-year,
density-profile-only sweep), but each of the 25 points now runs 100
model-years and keeps the full PrognosticState every ~month instead of a
reduced density profile every 30 days. Kept in its own folder, generated for
the wandb-sweep launch path (test/sweep/ck_ceps_100y.yaml +
sweep_ck_ceps_100y.py), so it doesn't mix with the 30-year sweep it's derived
from.
"""

import dataclasses
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mini_veros import loop
from mini_veros.setups.acc import full
from mini_veros.state import IntegratorState, PrognosticState, StatefulDiag, Tendencies

STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
DATA_DIR = STORE / "MiniVeros-Autodiff" / "results" / "ck_ceps_100y_sweep"

# dt_tracer = 43200s (12h) -> 2 steps/day, 730 steps/model-year (see test/ck_ceps_sweep/common.py).
STEPS_PER_DAY = int(round(86400.0 / 43200.0))
STEPS_PER_YEAR = STEPS_PER_DAY * 365
N_YEARS = 100
LOG_EVERY_DAYS = 30                         # ~1 month
LOG_EVERY = LOG_EVERY_DAYS * STEPS_PER_DAY  # 60 steps

def steps_for_years(years: float) -> int:
    """Nearest whole number of LOG_EVERY-step (monthly) snapshots to `years` model-years.

    100*730/60 = 1216.67, not integer (unlike the 30-year sweep, where 30*730/60 = 365
    exactly) -- round to the nearest whole number of monthly snapshots instead of
    asserting divisibility. Lands within 0.03% of the requested year count."""
    n_logs = round(years * STEPS_PER_YEAR / LOG_EVERY)
    return n_logs * LOG_EVERY


N_STEPS = steps_for_years(N_YEARS)          # 73020 steps = ~100.03 model-years
N_LOGS = N_STEPS // LOG_EVERY

C_K_DEFAULT = 0.10
C_EPS_DEFAULT = 0.70
# Wider than test/ck_ceps_sweep/common.py's factor-16, 5x5 grid: factor-64 range around
# each default, 10 points/axis, log2-spaced (100 combos total).
_GRID_EXPONENTS = tuple(-3.0 + 6.0 * i / 9 for i in range(10))
C_K_GRID = tuple(C_K_DEFAULT * 2.0**e for e in _GRID_EXPONENTS)
C_EPS_GRID = tuple(C_EPS_DEFAULT * 2.0**e for e in _GRID_EXPONENTS)


def full_state_log_select_fn(integrator_state):
    """log_select_fn that keeps the whole PrognosticState (u, v, temp, salt, tke, eke,
    psi) each logged step, instead of a reduced profile (see
    test/ck_ceps_sweep/run_top5_full_state.py, the 30-year/top-5-points precedent)."""
    return integrator_state.state


def run_one_point(c_k: float, c_eps: float, n_steps: int = N_STEPS, initial_integrator_state=None):
    """Build acc/full with (c_k, c_eps) overridden and run n_steps forward, logging the
    full state every LOG_EVERY steps. Starts from a cold start unless
    initial_integrator_state is given (see resume_one_point, which passes one loaded via
    load_checkpoint). Returns (zt, states, final_integrator_state): states is a
    PrognosticState whose leaves each carry a leading (n_steps // LOG_EVERY,) axis.
    Raises eqx.EquinoxRuntimeError if the run diverges (loop.run freezes the state and
    eqx.error_if raises on return)."""
    model, state0, forcing_fn = full.build({"c_k": c_k, "c_eps": c_eps})
    if initial_integrator_state is not None:
        state0 = initial_integrator_state
    run_fn = eqx.filter_jit(loop.run)
    final_integrator_state, states = run_fn(model, state0, forcing_fn, full_state_log_select_fn, n_steps, LOG_EVERY)
    return model.grid.zt, states, final_integrator_state


def resume_one_point(c_k: float, c_eps: float, checkpoint_path: Path, extra_years: float):
    """Continue a run from a save_checkpoint'd IntegratorState for `extra_years` more
    model-years, logging the full state every LOG_EVERY steps same as run_one_point.
    model/forcing_fn are rebuilt fresh via full.build inside run_one_point -- that's
    deterministic given the same (c_k, c_eps), so only the dynamic state (fields, AB2
    tendency history, carried density/warm-start diagnostics) needs to come from the
    checkpoint, not the model/grid/topology."""
    integrator_state = load_checkpoint(checkpoint_path)
    n_steps = steps_for_years(extra_years)
    return run_one_point(c_k, c_eps, n_steps=n_steps, initial_integrator_state=integrator_state)


def point_path(run_id: str, c_k: float, c_eps: float) -> Path:
    return DATA_DIR / "full_state" / run_id / f"ck{c_k:.4g}_eps{c_eps:.4g}.npz"


def checkpoint_path(run_id: str, c_k: float, c_eps: float) -> Path:
    return DATA_DIR / "checkpoints" / run_id / f"ck{c_k:.4g}_eps{c_eps:.4g}.npz"


def save_point(path: Path, c_k: float, c_eps: float, zt, states) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {name: np.asarray(getattr(states, name))
              for name in ("u", "v", "temp", "salt", "psi", "tke", "eke", "time")
              if getattr(states, name) is not None}
    np.savez(path, c_k=c_k, c_eps=c_eps, zt=np.asarray(zt),
              n_years=N_YEARS, n_steps=N_STEPS, log_every_days=LOG_EVERY_DAYS, **fields)


def _dump_dataclass(prefix: str, obj) -> dict:
    """np.savez-able dict of a state.py dataclass's non-None fields, keyed
    f"{prefix}__{field}" -- only the fields this run's config actually populates are
    written (e.g. dpsin only under enable_streamfunction), the rest reconstruct as None."""
    return {
        f"{prefix}__{f.name}": np.asarray(getattr(obj, f.name))
        for f in dataclasses.fields(obj) if getattr(obj, f.name) is not None
    }


def _load_dataclass(cls, prefix: str, npz):
    """Inverse of _dump_dataclass: rebuild `cls` from the f"{prefix}__*" keys in npz."""
    plen = len(prefix) + 2
    kwargs = {k[plen:]: jnp.asarray(v) for k, v in npz.items() if k.startswith(f"{prefix}__")}
    return cls(**kwargs)


def save_checkpoint(path: Path, c_k: float, c_eps: float, zt, final_integrator_state: IntegratorState) -> None:
    """Save the full IntegratorState (state + tendency_m1/m2 + statefuldiag_m1) needed to
    resume a run exactly where it left off. Unlike save_point's PrognosticState-only
    monthly snapshots, this also keeps the AB2 tendency history and the carried
    density/pressure-warm-start diagnostics that full.build's cold start otherwise
    re-zeros from t=0 initial conditions (setup.init_integrator_state) -- without them a
    resumed run's first step would blend the resumed fields against year-0 diagnostics
    instead of continuing smoothly. `tendency` itself (the top-level field) isn't saved:
    loop.run always returns it as None, recomputed fresh at the start of the next step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    S = final_integrator_state
    data = dict(c_k=c_k, c_eps=c_eps, zt=np.asarray(zt))
    data.update(_dump_dataclass("state", S.state))
    data.update(_dump_dataclass("tendency_m1", S.tendency_m1))
    data.update(_dump_dataclass("tendency_m2", S.tendency_m2))
    data.update(_dump_dataclass("statefuldiag_m1", S.statefuldiag_m1))
    np.savez(path, **data)


def load_checkpoint(path: Path) -> IntegratorState:
    """Rebuild the IntegratorState save_checkpoint wrote, ready to hand to run_one_point
    (via resume_one_point) to continue a run from exactly where it stopped."""
    with np.load(path) as npz:
        state = _load_dataclass(PrognosticState, "state", npz)
        tendency_m1 = _load_dataclass(Tendencies, "tendency_m1", npz)
        tendency_m2 = _load_dataclass(Tendencies, "tendency_m2", npz)
        statefuldiag_m1 = _load_dataclass(StatefulDiag, "statefuldiag_m1", npz)
    return IntegratorState(
        state=state, tendency=None, tendency_m1=tendency_m1, tendency_m2=tendency_m2,
        statefuldiag_m1=statefuldiag_m1,
    )
