"""
Diagnostic companion to fig_density_profile_top5.py: just the potential
density profiles as plain lines (no contourf, no inset), 3 panels --
whole-basin mean, the periodic ACC channel only (y<-20, south of the
"Atlantic block" landmass), and their difference (basin - channel) -- so the
profile shape itself can be checked in isolation, and so the channel proper
can be compared against the basin average it sits inside.

Writes report/ck_ceps_density_figures/fig_density_profile_lines.{pdf,png}.

    python test/ck_ceps_sweep/fig_density_profile_lines.py

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
    LINE_CMAP = cmocean.cm.deep_r
except ImportError:
    LINE_CMAP = style.SEQ_BLUE


def weighted_profile(rho, weight):
    """rho: (T, nx, ny, nz), weight: (nx, ny) -> (nz,) time- and area-weighted mean profile."""
    denom = weight.sum()
    return (rho * weight[None, :, :, None]).sum(axis=(1, 2)).mean(axis=0) / denom


def main():
    ref_model, _, _ = full.build({})
    zt = np.asarray(ref_model.grid.zt)
    y = np.asarray(ref_model.grid.yt[2:-2])
    area = np.asarray(ref_model.grid.area_t[2:-2, 2:-2])
    ocean_xy = np.asarray(ref_model.boundary_conditions.maskT[2:-2, 2:-2, :]).any(axis=2)
    basin_weight = area * ocean_xy
    channel_weight = basin_weight * (y < -20)[None, :]
    eq_of_state_type = ref_model.config.eq_of_state_type

    fig, axes = style.plt.subplots(1, 3, figsize=(15.5, 6.4), sharey=True)

    for n, (c_k, c_eps) in enumerate(common.TOP5):
        d = np.load(common.full_state_path(c_k, c_eps))
        temp = d["temp"][-common.KEEP_LAST:, 2:-2, 2:-2, :]
        salt = d["salt"][-common.KEEP_LAST:, 2:-2, 2:-2, :]
        rho = get_potential_rho(eq_of_state_type, salt, temp, 0.0)  # (KEEP_LAST, nx, ny, nz)

        basin_profile = weighted_profile(rho, basin_weight)
        channel_profile = weighted_profile(rho, channel_weight)
        diff_profile = basin_profile - channel_profile

        color = LINE_CMAP(0.15 + 0.7 * n / (len(common.TOP5) - 1))
        label = f"c_k={c_k:.3g}, c_eps={c_eps:.3g}"
        axes[0].plot(basin_profile, -zt, color=color, lw=2.0, marker="o", ms=3, label=label)
        axes[1].plot(channel_profile, -zt, color=color, lw=2.0, marker="o", ms=3, label=label)
        axes[2].plot(diff_profile, -zt, color=color, lw=2.0, marker="o", ms=3, label=label)

    titles = ["whole basin", "ACC channel only (y<-20)", "basin - ACC channel"]
    for ax, title in zip(axes, titles):
        ax.axvline(0.0, color=style.INK_MUTED, lw=1.0, zorder=0)
        ax.set_xlabel("potential density anomaly (kg/m^3)")
        ax.set_title(title)
        ax.grid(True, ls="--", alpha=0.5)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("depth (m)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("basin-mean potential density profile: whole basin vs ACC channel (diagnostic: lines only)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = common.FIG_DIR / "fig_density_profile_lines"
    common.FIG_DIR.mkdir(parents=True, exist_ok=True)
    style.save(fig, out_path)


if __name__ == "__main__":
    main()
