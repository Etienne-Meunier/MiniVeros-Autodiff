#!/usr/bin/env python3
"""
Investigation script: is `forc` (the RHS handed to the pressure solver) bit-for-bit
(0 ulp) identical between mini_veros and real veros at step 1, for acc_surface_pressure
(enable_streamfunction=False)? If not, exactly which upstream quantity is the first to
diverge?

Walks an ordered checkpoint chain -- static grid, initial density, forcing, step-1
tendency, integrated velocity, z-integrated transport, forc -- printing max abs diff
per checkpoint. Does not modify veros: real veros's own computed values are only read
(via wrapping its kernel functions to capture intermediates, never changing what they
return). mini_veros may need real source fixes as divergences are found and resolved --
that's expected; rerun this script after each fix to confirm the checkpoint clears and
find the next one.

Run: python test/investigate_forc_bitexact.py
"""

import dataclasses
import importlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))
VEROS_PATH = REPO_ROOT / "veros"
sys.path.insert(0, str(VEROS_PATH))

VARIANT_NAME = "acc_surface_pressure"
FAMILY = "acc_basic"
OVERRIDES = dict(enable_streamfunction=False)

results = []  # (name, max_abs_diff, extra_info_str)


def check(name, real_val, mini_val):
    a = np.asarray(real_val, dtype=np.float64)
    b = np.asarray(mini_val, dtype=np.float64)
    if a.shape != b.shape:
        results.append((name, float("nan"), f"SHAPE MISMATCH real={a.shape} mini={b.shape}"))
        print(f"  [{'SHAPE MISMATCH':>9s}] {name:20s} real={a.shape} mini={b.shape}")
        return
    d = np.abs(a - b)
    m = d.max() if d.size else 0.0
    extra = ""
    if m > 0:
        loc = np.unravel_index(np.argmax(d), d.shape)
        nz = np.count_nonzero(d)
        extra = f"argmax@{loc} real={a[loc]:.6g} mini={b[loc]:.6g}  nonzero={nz}/{d.size}"
    results.append((name, float(m), extra))
    tag = "PASS" if m == 0 else "FAIL"
    print(f"  [{tag:>9s}] {name:20s} max_abs_diff={m:.3e}  {extra}")


# ============================================================= REAL VEROS =============================================================
print("=== building real veros ===")
vsp = importlib.import_module("veros.core.external.solve_pressure")
external_pkg = importlib.import_module("veros.core.external")
momentum_mod = importlib.import_module("veros.core.momentum")

captured_real = {}

orig_prepare_forcing = vsp.prepare_forcing
def wrapped_prepare_forcing(state):
    vs = state.variables
    # capture rho as prepare_forcing itself reads it (rho[..., tau], not yet
    # touched by thermodynamics this step -- momentum() runs before
    # thermodynamics() in veros.py's step order) -- reading it any later (e.g.
    # after the step completes) picks up a post-thermodynamics/rotated value
    # instead, comparing apples to oranges against mini's pre-step rho.
    captured_real["rho_used"] = np.asarray(vs.rho[:, :, :, vs.tau])
    state_update, forc = orig_prepare_forcing(state)
    # state_update.u/v[...,taup1] here are prepare_forcing's own integrated-but-
    # not-yet-pressure-corrected values -- the true counterpart to mini's u1/v1.
    # Reading vs.u[...,taup1] later (after the full step) would instead pick up
    # the FINAL value, since solve_pressure's barotropic_velocity_update does a
    # second in-place update_add onto the same taup1 slot right after this.
    captured_real["u1"] = np.asarray(state_update.u[..., vs.taup1])
    captured_real["v1"] = np.asarray(state_update.v[..., vs.taup1])
    captured_real["forc"] = np.asarray(forc)
    captured_real["du_final"] = np.asarray(state_update.du[..., vs.tau])
    captured_real["dv_final"] = np.asarray(state_update.dv[..., vs.tau])
    captured_real["du_mix"] = np.asarray(vs.du_mix)
    captured_real["dv_mix"] = np.asarray(vs.dv_mix)
    return state_update, forc
vsp.prepare_forcing = wrapped_prepare_forcing

from variant_util import build_real_variant, build_mini_variant, FAMILIES, _route_overrides  # noqa: E402

sim, si_r, sf_r, _, ts_r, rec_r = build_real_variant(
    VARIANT_NAME, FAMILY, OVERRIDES, 1, VEROS_PATH, record_interval=None
)
vs = sim.state.variables
settings = sim.state.settings

# ============================================================= MINI VEROS =============================================================
print("\n=== building mini_veros ===")
sys.path.insert(0, str(REPO_ROOT / "mini-veros"))
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
from mini_veros import loop  # noqa: E402
from mini_veros.core.external import solve_pressure as mini_vsp  # noqa: E402
from mini_veros.core.external.solvers import scipy_jax as mini_scipy_jax  # noqa: E402
from mini_veros.state import Tendencies  # noqa: E402

captured_mini = {}

orig_elliptic_solve = mini_scipy_jax.elliptic_solve
def mini_elliptic_wrapped(model, rhs, x0, boundary_val=None):
    captured_mini["forc"] = np.asarray(rhs)
    captured_mini["x0"] = np.asarray(x0)
    return orig_elliptic_solve(model, rhs, x0, boundary_val)
mini_scipy_jax.elliptic_solve = mini_elliptic_wrapped
mini_vsp.scipy_jax.elliptic_solve = mini_elliptic_wrapped

orig_mini_solve_pressure = mini_vsp.solve_pressure
def mini_solve_pressure_wrapped(model, S, dS, dSm1, statefuldiag_m1):
    # capture pre-solve intermediates the same way solve_pressure computes them,
    # without changing anything -- mirrors the function body up to elliptic_solve
    import jax.numpy as jnp
    settings = model.config
    if (
        settings.enable_implicit_vert_friction
        or settings.enable_explicit_vert_friction
        or settings.enable_ray_friction
        or settings.enable_bottom_friction
        or settings.enable_quadratic_bottom_friction
        or settings.enable_momentum_sources
    ):
        du_mix, dv_mix = dS.du_mix, dS.dv_mix
    else:
        du_mix, dv_mix = jnp.zeros_like(dS.du), jnp.zeros_like(dS.dv)
    captured_mini["du_final"] = np.asarray(dS.du)
    captured_mini["dv_final"] = np.asarray(dS.dv)
    captured_mini["du_mix"] = np.asarray(du_mix)
    captured_mini["dv_mix"] = np.asarray(dv_mix)

    dt_mom = settings.dt_mom
    eps = model.parameters.AB_eps
    grid, bc = model.grid, model.boundary_conditions
    u1 = S.u + dt_mom * (du_mix + (1.5 + eps) * dS.du - (0.5 + eps) * dSm1.du) * bc.maskU
    v1 = S.v + dt_mom * (dv_mix + (1.5 + eps) * dS.dv - (0.5 + eps) * dSm1.dv) * bc.maskV
    captured_mini["u1"] = np.asarray(u1)
    captured_mini["v1"] = np.asarray(v1)

    uloc = jnp.zeros_like(bc.hu)
    vloc = jnp.zeros_like(bc.hv)
    uloc = uloc.at[2:-2, 2:-2].set(jnp.sum(u1[2:-2, 2:-2, :] * bc.maskU[2:-2, 2:-2, :] * grid.dzt, axis=2) / dt_mom)
    vloc = vloc.at[2:-2, 2:-2].set(jnp.sum(v1[2:-2, 2:-2, :] * bc.maskV[2:-2, 2:-2, :] * grid.dzt, axis=2) / dt_mom)
    from mini_veros.core import utilities
    uloc = utilities.enforce_boundaries(uloc, settings.enable_cyclic_x)
    vloc = utilities.enforce_boundaries(vloc, settings.enable_cyclic_x)
    captured_mini["uloc"] = np.asarray(uloc)
    captured_mini["vloc"] = np.asarray(vloc)

    return orig_mini_solve_pressure(model, S, dS, dSm1, statefuldiag_m1)
mini_vsp.solve_pressure = mini_solve_pressure_wrapped
loop.solve_pressure.solve_pressure = mini_solve_pressure_wrapped

# built and stepped eagerly (not jax.jit'd like build_mini_variant) so the capture
# wrappers above -- which call np.asarray on intermediates -- work during tracing
setup_mod = importlib.import_module(FAMILIES[FAMILY]["mini_module"])
model, step0, forcing_fn = setup_mod.build()
model = _route_overrides(model, OVERRIDES)
nisle = model.boundary_conditions.psin.shape[-1]
zero_tend = Tendencies.init(model.config, nisle)
step0 = dataclasses.replace(step0, tendency_m1=zero_tend, tendency_m2=zero_tend)
grid, bc, params = model.grid, model.boundary_conditions, model.parameters

mini_force = forcing_fn(model, step0.state)
step1 = loop.step(model, step0, mini_force)

# ============================================================= CHECKPOINTS =============================================================
print("\n=== checkpoint chain (real veros vs mini_veros) ===")

print("-- static grid --")
check("yt", vs.yt, grid.yt)
check("yu", vs.yu, grid.yu)
check("cost", vs.cost, grid.cost)
check("cosu", vs.cosu, grid.cosu)
check("tantr", vs.tantr, grid.tantr)
check("dxt", vs.dxt, grid.dxt)
check("dxu", vs.dxu, grid.dxu)
check("dyt", vs.dyt, grid.dyt)
check("dyu", vs.dyu, grid.dyu)
check("dzt", vs.dzt, grid.dzt)
check("dzw", vs.dzw, grid.dzw)
check("area_t", vs.area_t, grid.area_t)
check("area_u", vs.area_u, grid.area_u)
check("area_v", vs.area_v, grid.area_v)
check("coriolis_t", vs.coriolis_t, grid.coriolis_t)
check("beta", vs.beta, grid.beta)
check("hu", vs.hu, bc.hu)
check("hv", vs.hv, bc.hv)
check("ht", vs.ht, bc.ht)
check("maskT[-1]", vs.maskT[:, :, -1], bc.maskT[:, :, -1])
check("maskU[-1]", vs.maskU[:, :, -1], bc.maskU[:, :, -1])
check("maskV[-1]", vs.maskV[:, :, -1], bc.maskV[:, :, -1])

print("-- initial density (as read by prepare_forcing / tend_hydrostatic_pressure) --")
check("rho(t=0)", captured_real["rho_used"], step0.statefuldiag_m1.rho)

print("-- forcing --")
check("surface_taux", vs.surface_taux, mini_force.surface_taux)
check("surface_tauy", vs.surface_tauy, mini_force.surface_tauy)

print("-- step-1 pre-solve tendency (coriolis/advection/friction are exactly 0 here since u=v=0; isolates wind+hydro) --")
check("du_final", captured_real["du_final"], captured_mini["du_final"])
check("dv_final", captured_real["dv_final"], captured_mini["dv_final"])
check("du_mix", captured_real["du_mix"], captured_mini["du_mix"])
check("dv_mix", captured_real["dv_mix"], captured_mini["dv_mix"])

print("-- integrated velocity (u1/v1, pre pressure-gradient correction) --")
check("u1", captured_real["u1"], captured_mini["u1"])
check("v1", captured_real["v1"], captured_mini["v1"])

print("-- forc --")
check("forc", captured_real["forc"], captured_mini["forc"])

# ============================================================= SUMMARY =============================================================
print("\n=== summary ===")
print("\n=== location lists: du_final vs forc (are they the same cells?) ===")
maskT_np = np.asarray(bc.maskT[:, :, -1])
def locs(name, real_val, mini_val, is3d):
    a, b = np.asarray(real_val, dtype=np.float64), np.asarray(mini_val, dtype=np.float64)
    d = np.abs(a - b)
    idx = np.argwhere(d > 0)
    print(f"-- {name}: {len(idx)} nonzero --")
    for loc in idx:
        loc = tuple(loc)
        xy = loc[:2]
        w = maskT_np[xy[0] - 1, xy[1]]
        e = maskT_np[xy[0] + 1, xy[1]]
        print(f"   {loc}  diff={d[loc]:.3e}  maskT(W,E)=({w:.0f},{e:.0f})")

locs("du_final", captured_real["du_final"], captured_mini["du_final"], True)
locs("forc", captured_real["forc"], captured_mini["forc"], False)

first_fail = next((r for r in results if r[1] != 0.0 and not np.isnan(r[1])), None)
if first_fail is None:
    print("all checkpoints PASS -- forc is bit-identical.")
else:
    print(f"FIRST DIVERGENCE: {first_fail[0]}  max_abs_diff={first_fail[1]:.3e}  {first_fail[2]}")
    n_after = sum(1 for r in results if r[1] != 0.0 and not np.isnan(r[1]))
    print(f"{n_after}/{len(results)} checkpoints nonzero total (propagation from there on is expected, not independently informative).")
