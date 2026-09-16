"""
Reads the 5 full-state runs and writes report/ck_ceps_density_figures/
fig_density_profile_top5.{pdf,png}: 5 side-by-side panels, one per run, each
a latitude-depth section (contourf) of the zonal-mean potential density
(final-model-year average). Design (cmocean deep_r, black contour lines,
light-gray land/background) after DINO-Fusion's
Results/notebooks/ComputeDensity.ipynb (Results/figures/
states_after_10y_integration.png) -- minus that figure's 1D inset, which
kept causing rendering artifacts (transparency bleed-through from the main
panel's contour lines) without adding information fig_density_profile_lines.py
doesn't already show cleanly on its own; see that script instead for the
basin-mean profile by itself.

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

style.use()

try:
    import cmocean
    CMAP = cmocean.cm.deep_r
except ImportError:
    CMAP = style.SEQ_BLUE

LEVELS = np.linspace(-0.6, 1.2, 13)


def main():
    ref_model, _, _ = full.build({})
    y = np.asarray(ref_model.grid.yt[2:-2])
    zt = np.asarray(ref_model.grid.zt)
    ocean_xy = np.asarray(ref_model.boundary_conditions.maskT[2:-2, 2:-2, :]).any(axis=2)  # (nx, ny)
    eq_of_state_type = ref_model.config.eq_of_state_type

    fig, axes = style.plt.subplots(1, 5, figsize=(17.5, 4.6), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.93, top=0.85, bottom=0.12, wspace=0.35)
    cax = fig.add_axes([0.955, 0.15, 0.012, 0.62])

    mesh = None
    for ax, (c_k, c_eps) in zip(axes, common.TOP5):
        d = np.load(common.full_state_path(c_k, c_eps))
        temp = d["temp"][-common.KEEP_LAST:, 2:-2, 2:-2, :].mean(axis=0)
        salt = d["salt"][-common.KEEP_LAST:, 2:-2, 2:-2, :].mean(axis=0)
        rho = get_potential_rho(eq_of_state_type, salt, temp, 0.0)  # (nx, ny, nz)

        n_ocean_x = np.maximum(ocean_xy.sum(axis=0), 1)  # (ny,) -- dxt is uniform in x, so an
        zonal_mean = (rho * ocean_xy[:, :, None]).sum(axis=0) / n_ocean_x[:, None]  # (ny, nz) unweighted
        # count-mean along a fixed y is already the correct area-weighted zonal mean.

        ax.set_facecolor(style.LAND)
        mesh = ax.contourf(y, -zt, zonal_mean.T, levels=LEVELS, cmap=CMAP, extend="both")
        cs = ax.contour(y, -zt, zonal_mean.T, levels=LEVELS, colors="k", linewidths=0.6)
        ax.clabel(cs, fmt="%.1f", fontsize=6.5, inline=True)
        ax.axvline(-20, color="gray", ls="--", lw=1.0)
        ax.set_title(f"c_k={c_k:.3g}, c_eps={c_eps:.3g}", fontsize=10)
        ax.set_xlabel("latitude")
        ax.invert_yaxis()

    axes[0].set_ylabel("depth (m)")
    fig.colorbar(mesh, cax=cax, label="potential density anomaly (kg/m^3, rel. rho0=1024)")
    fig.suptitle("zonal-mean potential density, final model year", y=0.97)

    out_path = common.FIG_DIR / "fig_density_profile_top5"
    common.FIG_DIR.mkdir(parents=True, exist_ok=True)
    style.save(fig, out_path)


if __name__ == "__main__":
    main()
