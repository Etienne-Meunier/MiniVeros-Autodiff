"""
Companion to fig_density_profile_top5.py: same latitude-depth zonal-mean
potential density sections, but each panel shows that run's field minus the
5-run mean field (diverging colormap, scaled to the largest anomaly) --
same idea as fig_mld_snapshots.py's MLD anomaly-from-mean map.

Writes report/ck_ceps_density_figures/fig_density_section_anomaly.{pdf,png}.

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


def main():
    ref_model, _, _ = full.build({})
    y = np.asarray(ref_model.grid.yt[2:-2])
    zt = np.asarray(ref_model.grid.zt)
    ocean_xy = np.asarray(ref_model.boundary_conditions.maskT[2:-2, 2:-2, :]).any(axis=2)
    n_ocean_x = np.maximum(ocean_xy.sum(axis=0), 1)
    eq_of_state_type = ref_model.config.eq_of_state_type

    zonal_means = []
    for c_k, c_eps in common.TOP5:
        d = np.load(common.full_state_path(c_k, c_eps))
        temp = d["temp"][-common.KEEP_LAST:, 2:-2, 2:-2, :].mean(axis=0)
        salt = d["salt"][-common.KEEP_LAST:, 2:-2, 2:-2, :].mean(axis=0)
        rho = get_potential_rho(eq_of_state_type, salt, temp, 0.0)  # (nx, ny, nz)
        zonal_means.append((rho * ocean_xy[:, :, None]).sum(axis=0) / n_ocean_x[:, None])  # (ny, nz)

    reference = np.mean(np.stack(zonal_means), axis=0)  # (ny, nz), 5-run mean
    anomalies = [zm - reference for zm in zonal_means]

    vmax = max(np.abs(a).max() for a in anomalies)
    norm = style.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = style.plt.subplots(1, 5, figsize=(17.5, 4.6), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.93, top=0.85, bottom=0.12, wspace=0.35)
    cax = fig.add_axes([0.955, 0.15, 0.012, 0.62])

    mesh = None
    for ax, anomaly, (c_k, c_eps) in zip(axes, anomalies, common.TOP5):
        ax.set_facecolor(style.LAND)
        mesh = ax.contourf(y, -zt, anomaly.T, levels=15, cmap=style.DIVERGING, norm=norm, extend="both")
        ax.contour(y, -zt, anomaly.T, levels=[0.0], colors="k", linewidths=0.8)
        ax.axvline(-20, color="gray", ls="--", lw=1.0)
        ax.set_title(f"c_k={c_k:.3g}, c_eps={c_eps:.3g}", fontsize=10)
        ax.set_xlabel("latitude")
        ax.invert_yaxis()

    axes[0].set_ylabel("depth (m)")
    fig.colorbar(mesh, cax=cax, label="potential density anomaly from 5-run mean (kg/m^3)")
    fig.suptitle("zonal-mean potential density, minus the 5-run mean, final model year", y=0.97)

    out_path = common.FIG_DIR / "fig_density_section_anomaly"
    common.FIG_DIR.mkdir(parents=True, exist_ok=True)
    style.save(fig, out_path)


if __name__ == "__main__":
    main()
