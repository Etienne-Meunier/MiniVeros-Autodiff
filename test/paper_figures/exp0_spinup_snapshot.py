"""
Experiment 0 -- snapshot of the spun-up state: mixed-layer depth and vertically
integrated kinetic energy.

Reuses the cached 30-year spin-up (no new simulation), member 0's settled
state at the point every other experiment in this paper starts from.

MLD: depth where potential density first exceeds the surface value by
0.03 kg/m^3 (the de Boyer Montegut et al. 2004 density criterion), scanning
down from the surface; a column that never crosses the threshold is reported
as fully mixed to the bottom (same definition as
test/ck_ceps_sweep/fig_mld_snapshots.py).

Vertically integrated kinetic energy: sum_z (u^2 + v^2) * dz, with u and v
first interpolated from their native Arakawa-C staggered locations (east and
north cell faces) onto the T-point grid via mini_veros.core.numerics'
ugrid_to_tgrid/vgrid_to_tgrid, so the two components are co-located before
summing -- pairing u and v at their raw, unaligned grid indices would mix
velocities from physically different points.

    python test/paper_figures/exp0_spinup_snapshot.py
"""

import numpy as np

import common
from mini_veros.core.density.get_rho import get_potential_rho
from mini_veros.core.numerics import ugrid_to_tgrid, vgrid_to_tgrid

MLD_THRESHOLD = 0.03  # kg/m^3, de Boyer Montegut et al. 2004 density criterion


def mld_from_density(rho, zt):
    """rho: (..., nz) potential density, index 0=bottom .. nz-1=surface (model convention).
    Returns (...,) MLD depth in positive meters.
    """
    rev = rho[..., ::-1]  # index 0=surface .. nz-1=bottom
    zt_rev = zt[::-1]
    exceed = (rev - rev[..., :1]) >= MLD_THRESHOLD
    any_exceed = exceed.any(axis=-1)
    first_idx = np.where(any_exceed, np.argmax(exceed, axis=-1), len(zt) - 1)
    return -zt_rev[first_idx]


def main():
    model, state0, _ = common.spinup()
    state = state0.state

    rho = get_potential_rho(model.config.eq_of_state_type, state.salt, state.temp, 0.0)
    zt = np.asarray(model.grid.zt)
    mld = mld_from_density(np.asarray(common.interior(rho)), zt)  # (nx, ny)

    u_t, v_t = ugrid_to_tgrid(model, state.u), vgrid_to_tgrid(model, state.v)
    dzt = model.grid.dzt
    ke = np.asarray(common.interior((u_t**2 + v_t**2) * dzt[None, None, :]).sum(axis=-1))  # (nx, ny)

    x, y, land = common.grid_arrays(model)
    xu = np.asarray(model.grid.xu[2:-2])  # east-face longitude of each interior T-cell -- its real cell edge
    yu = np.asarray(model.grid.yu[2:-2])  # north-face latitude of each interior T-cell -- its real cell edge
    common.save("exp0_spinup_snapshot", x=x, y=y, land=land, mld=mld, ke=ke, xu=xu, yu=yu)


if __name__ == "__main__":
    main()
