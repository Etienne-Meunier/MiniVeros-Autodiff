#!/usr/bin/env python3
"""
Regression tests for the three causes behind report/divergence_report.md.

These do NOT gate long-horizon pointwise agreement -- report/matrix_report.md
already shows that is unreachable, and investigate_divergence.py's `twin`
experiment shows why (real veros diverges from itself the same way given an
equally small perturbation). What is gated here is everything that happens
*before* chaos takes over, which is where a genuine port bug would show:

  * step-0 parity: mini_veros and veros must start from the same state.
  * physics parity: with the elliptic solver's stopping rule tightened out of
    the way, a handful of steps must agree near float64 roundoff, for every
    physics option in the matrix.
  * the seed itself: the first-step gap must actually be the solver's
    tolerance, i.e. it must shrink by orders of magnitude when that
    tolerance is tightened. If it stops doing that, something else has
    become the dominant error source and this whole story needs revisiting.

Run with:
    pytest test/test_divergence.py                     # acc variants
    MINIVEROS_TEST_GLOBAL=1 pytest test/test_divergence.py   # + global (slow)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from investigate_divergence import _normalized_diffs
from setups_matrix import FAMILIES, VARIANTS, VARIANTS_BY_NAME
from variant_util import DEFAULT_SOLVER_ATOL, TIGHT_SOLVER_ATOL, forced_solver_atol
from util import configure_veros_runtime

VEROS_PATH = REPO_ROOT / "veros"

# Roundoff-scale bound for a few steps at a tight solver tolerance. Well above
# float64 eps because the setups' own grid/forcing construction differs by
# numpy-vs-jax evaluation order (see report/error_math.md), and because a few
# steps of the flow amplify that.
#
# global_4deg gets a looser one: unlike acc it starts with a real barotropic
# mode, and building it runs streamfunction_init's own elliptic solve at the
# shipped atol=1e-8 -- worth ~1.6e-7 normalized before a single step runs.
# Tightening the per-step solver does not remove that; it is baked into the
# initial state both here and in `test_initial_state_parity`.
TIGHT_TOLERANCE_BOUND = {"acc": 1e-8, "global": 1e-6}
INITIAL_PARITY_BOUND = {"acc": 1e-12, "global": 1e-6}

# Bound for variants advecting TKE/EKE with the superbee limiter. The limiter
# is discontinuous (`where(vel > 0)`, `clip`, and the `abs(rj) < 1e-20` guard),
# so a roundoff-level input difference produces a finite flux difference. This
# is veros's own behaviour, not a port artifact: running real veros against
# itself from a 1e-15 relative temp perturbation produces the same ~3e-7 tke
# jump at the same step (investigate_divergence.py `twin`).
SUPERBEE_BOUND = 1e-2

_SUPERBEE_TKE = "enable_tke_superbee_advection"

STEPS = int(os.environ.get("MINIVEROS_TEST_STEPS", 5))
RUN_GLOBAL = os.environ.get("MINIVEROS_TEST_GLOBAL", "") not in ("", "0", "false")


def _variants(group):
    return [v["name"] for v in VARIANTS if FAMILIES[v["family"]]["group"] == group]


ACC_VARIANTS = _variants("acc")
GLOBAL_VARIANTS = _variants("global")
ALL_VARIANTS = ACC_VARIANTS + (GLOBAL_VARIANTS if RUN_GLOBAL else [])


@pytest.fixture(scope="session", autouse=True)
def _veros_runtime():
    configure_veros_runtime(VEROS_PATH)
    sys.path.insert(0, str(REPO_ROOT / "mini-veros"))


def _run_pair(name, n_steps, record_interval=1):
    from variant_util import build_mini_variant, build_real_variant

    variant = VARIANTS_BY_NAME[name]
    args = (variant["name"], variant["family"], variant["overrides"], n_steps, VEROS_PATH, record_interval)
    _, _, _, _, timesteps, mini_states = build_mini_variant(*args)
    _, _, _, _, _, real_states = build_real_variant(*args)
    return timesteps, mini_states, real_states


def _group_of(name):
    return FAMILIES[VARIANTS_BY_NAME[name]["family"]]["group"]


def _bound_for(name):
    overrides = VARIANTS_BY_NAME[name]["overrides"]
    if overrides.get(_SUPERBEE_TKE):
        return SUPERBEE_BOUND
    return TIGHT_TOLERANCE_BOUND[_group_of(name)]


@pytest.mark.parametrize("name", ALL_VARIANTS)
def test_initial_state_parity(name):
    """
    mini_veros and veros must agree at step 0, before a single step runs.

    This is what caught the surface-pressure init bug: mini_veros used to run
    an initial solve_pressure that real veros never runs (veros only calls
    external.streamfunction_init, and only under enable_streamfunction), so
    global_surface_pressure started with psi ~ O(10) and u/v ~ O(1e-2) where
    veros had exactly zero. No agreement downstream can recover from that.
    """
    from variant_util import build_mini_variant, build_real_variant

    variant = VARIANTS_BY_NAME[name]
    args = (variant["name"], variant["family"], variant["overrides"], 0, VEROS_PATH)
    _, mini_s0, *_ = build_mini_variant(*args)
    _, real_s0, *_ = build_real_variant(*args)

    diffs = _normalized_diffs(mini_s0, real_s0)
    worst_field = max(diffs, key=diffs.get)
    # the streamfunction init is itself an elliptic solve at the shipped
    # atol=1e-8, so global setups (which start with a real barotropic mode)
    # legitimately differ there; acc starts from rest and must be exact
    bound = INITIAL_PARITY_BOUND[_group_of(name)]
    assert diffs[worst_field] < bound, (
        f"{name}: step-0 mismatch in {worst_field} = {diffs[worst_field]:.3e} (bound {bound:.0e})"
    )


@pytest.mark.parametrize("name", ALL_VARIANTS)
def test_physics_matches_at_tight_solver_tolerance(name):
    """
    With the elliptic solver's stopping rule tightened, every physics option
    in the matrix must reproduce veros to near roundoff over a few steps.

    The default rule -- bicgstab(tol=0, atol=1e-8), the same call in both
    codes -- is an *absolute* residual bound, so it leaves the two solvers
    free to stop at points that differ by far more than float64 roundoff.
    Tightening it is what makes a physics comparison possible at all.
    """
    with forced_solver_atol(TIGHT_SOLVER_ATOL):
        _, mini_states, real_states = _run_pair(name, STEPS)

    diffs = _normalized_diffs(mini_states[-1], real_states[-1])
    worst_field = max(diffs, key=diffs.get)
    bound = _bound_for(name)
    assert diffs[worst_field] < bound, (
        f"{name}: after {STEPS} steps at atol={TIGHT_SOLVER_ATOL:g}, "
        f"{worst_field} differs by {diffs[worst_field]:.3e} (bound {bound:.0e})"
    )


def test_solver_tolerance_dominates_the_first_step():
    """
    The first-step gap must be set by the solver's stopping rule, not by the
    physics: tightening atol from 1e-8 to 1e-14 has to shrink it by orders of
    magnitude. If this ever stops holding, the seed of the long-horizon
    divergence has moved somewhere else and report/divergence_report.md's
    conclusion no longer follows.
    """
    name = "acc_basic"

    with forced_solver_atol(DEFAULT_SOLVER_ATOL):
        _, mini_loose, real_loose = _run_pair(name, 1)
    with forced_solver_atol(TIGHT_SOLVER_ATOL):
        _, mini_tight, real_tight = _run_pair(name, 1)

    loose = _normalized_diffs(mini_loose[-1], real_loose[-1])
    tight = _normalized_diffs(mini_tight[-1], real_tight[-1])

    assert loose["psi"] > 1e-9, f"expected a solver-tolerance-sized psi gap, got {loose['psi']:.3e}"
    assert tight["psi"] < loose["psi"] / 1e4, (
        f"tightening the solver barely helped: psi {loose['psi']:.3e} -> {tight['psi']:.3e}; "
        "the first-step gap is no longer dominated by the elliptic solve"
    )


@pytest.mark.skipif(not RUN_GLOBAL, reason="set MINIVEROS_TEST_GLOBAL=1 (slow)")
def test_surface_pressure_init_is_not_run():
    """
    Direct regression test on the fixed bug: with enable_streamfunction=False,
    mini_veros must not run an initial barotropic solve, because veros
    doesn't (VerosSetup.setup guards streamfunction_init on that setting).
    """
    from variant_util import build_mini_variant, build_real_variant

    variant = VARIANTS_BY_NAME["global_surface_pressure"]
    args = (variant["name"], variant["family"], variant["overrides"], 0, VEROS_PATH)
    _, mini_s0, *_ = build_mini_variant(*args)
    _, real_s0, *_ = build_real_variant(*args)

    for field in ("psi", "u", "v"):
        assert np.array_equal(mini_s0[field], real_s0[field]), (
            f"global_surface_pressure: {field} differs at step 0 "
            f"(mini max |{field}| = {np.nanmax(np.abs(mini_s0[field])):.3e}, veros starts from rest)"
        )
