"""
Reads the 5 full-state runs and writes report/ck_ceps_density_figures/
fig_basin_mld_time.{pdf,png}: basin-mean MLD (area-weighted, ocean surface
cells, same criterion as fig_mld_snapshots.py) through the 30-year run, all
5 runs overlaid -- Section 4's time view of the (c_k, c_eps) split.

Never re-runs a simulation.
"""

import sys
from pathlib import Path

import numpy as np

import common

sys.path.append(str(Path(__file__).resolve().parents[1] / "paper_figures"))
import style  # noqa: E402
from mini_veros.core.density.get_rho import get_potential_rho
from mini_veros.setups.acc import full

from fig_mld_snapshots import mld_from_density

style.use()


def main():
    ref_model, _, _ = full.build({})
    zt = np.asarray(ref_model.grid.zt)
    area = np.asarray(ref_model.grid.area_t[2:-2, 2:-2])
    ocean3d = np.asarray(ref_model.boundary_conditions.maskT[2:-2, 2:-2, :]) != 0
    surf_weight = area * ocean3d[:, :, -1]
    surf_denom = surf_weight.sum()
    eq_of_state_type = ref_model.config.eq_of_state_type

    fig, ax = style.plt.subplots(figsize=(7.5, 3.8))
    for n, (c_k, c_eps) in enumerate(common.TOP5):
        d = np.load(common.full_state_path(c_k, c_eps))
        temp, salt = d["temp"][:, 2:-2, 2:-2, :], d["salt"][:, 2:-2, 2:-2, :]
        years = (np.arange(temp.shape[0]) + 1) * common.LOG_EVERY_DAYS / 365.0

        rho = get_potential_rho(eq_of_state_type, salt, temp, 0.0)
        mld = mld_from_density(rho, zt)
        basin_mld = (mld * surf_weight[None]).sum(axis=(1, 2)) / surf_denom

        color = style.SERIES[n]
        std = common.rolling_std(basin_mld)
        ax.fill_between(years, basin_mld - std, basin_mld + std, color=color, alpha=0.25, lw=0, zorder=0)
        ax.plot(years, basin_mld, color=color, lw=1.3, label=f"c_k={c_k:.3g}, c_eps={c_eps:.3g}", zorder=1)

    ax.set_ylabel("basin-mean MLD (m)")
    ax.set_xlabel("model year")
    ax.grid(True, color=style.GRID, lw=0.6)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("basin-mean MLD through time")
    fig.tight_layout()

    out_path = common.FIG_DIR / "fig_basin_mld_time"
    common.FIG_DIR.mkdir(parents=True, exist_ok=True)
    style.save(fig, out_path)


if __name__ == "__main__":
    main()
