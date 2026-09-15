"""
Writes report/ck_ceps_density_figures/fig_setup.png: 3 panels describing the
acc/full setup (Section 1 of top5_runs_report.md) -- topography, the two
y-only forcing profiles, and the initial temperature profile. Pure setup,
no simulation output.

    python test/ck_ceps_sweep/fig_setup.py
"""

import sys
from pathlib import Path

import numpy as np

import common

sys.path.append(str(Path(__file__).resolve().parents[1] / "paper_figures"))
import style  # noqa: E402
from mini_veros.setups.acc import full

style.use()


def main():
    model, state0, _ = full.build({})
    grid, bc = model.grid, model.boundary_conditions
    x, y = np.asarray(grid.xt[2:-2]), np.asarray(grid.yt[2:-2])
    land = np.asarray(bc.maskT[2:-2, 2:-2, -1]) == 0

    # Same formulas as acc/common.py:build_forcing_fn (y-only, no time dependence).
    yu, yt = np.asarray(grid.yu), np.asarray(grid.yt)
    yt_min, yu_min, yt_max, yu_max = yt.min(), yu.min(), yt.max(), yu.max()
    taux = np.zeros_like(yt)
    taux = np.where(yt < -20, 0.1 * np.sin(np.pi * (yu - yu_min) / (-20.0 - yt_min)), taux)
    taux = np.where(yt > 10, 0.1 * (1 - np.cos(2 * np.pi * (yu - 10.0) / (yu_max - 10.0))), taux)
    t_star = np.full_like(yt, 15.0)
    t_star = np.where(yt < -20, 15 * (yt - yt_min) / (-20 - yt_min), t_star)
    t_star = np.where(yt > 20, 15 * (1 - (yt - 20) / (yt_max - 20)), t_star)

    zt = np.asarray(grid.zt)
    temp0 = np.asarray(state0.state.temp[5, 5, :])  # any ocean column -- horizontally uniform IC

    fig, axes = style.plt.subplots(1, 3, figsize=(13.5, 4.2))

    ocean_land_cmap = style.ListedColormap(["#dce6f0", style.LAND])
    axes[0].pcolormesh(x, y, land.T.astype(float), cmap=ocean_land_cmap, vmin=0, vmax=1)
    axes[0].axhline(-20, color=style.SERIES[1], ls="--", lw=1.3, label="y=-20 (channel opens)")
    axes[0].set_title("topography")
    axes[0].set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_aspect("equal")

    ax2 = axes[1]
    l1, = ax2.plot(yu, taux, color=style.SERIES[0], lw=1.6, label="tau_x(y)")
    ax2.set_xlabel("latitude")
    ax2.set_ylabel("wind stress tau_x (N/m^2)", color=style.SERIES[0])
    ax2.tick_params(axis="y", colors=style.SERIES[0])
    ax3 = ax2.twinx()
    l2, = ax3.plot(yt, t_star, color=style.SERIES[1], lw=1.6, label="t_star(y)")
    ax3.set_ylabel("restoring SST t_star (degC)", color=style.SERIES[1])
    ax3.tick_params(axis="y", colors=style.SERIES[1])
    ax2.set_title("forcing profiles (static, y-only)")
    ax2.legend(handles=[l1, l2], fontsize=8, loc="lower right")
    ax2.grid(True, color=style.GRID, lw=0.6)

    axes[2].plot(temp0, zt, color=style.SERIES[0], lw=1.8)
    axes[2].set_xlabel("initial temperature (degC)")
    axes[2].set_ylabel("depth z (m)")
    axes[2].set_title("initial condition")
    axes[2].grid(True, color=style.GRID, lw=0.6)

    fig.tight_layout()
    out_path = common.FIG_DIR / "fig_setup"
    common.FIG_DIR.mkdir(parents=True, exist_ok=True)
    style.save(fig, out_path)


if __name__ == "__main__":
    main()
