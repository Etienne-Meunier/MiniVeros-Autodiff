"""
Reads the 5 full-state runs and writes report/ck_ceps_density_figures/
fig_density_rms.{pdf,png}: step-to-step density change
RMS(rho[t+1]-rho[t]) over the basin (volume-weighted), log-scale, all 5 runs
overlaid -- the first convergence check (Section 3): falling toward a noise
floor is what "the density field is converging" looks like quantitatively.

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
    area = np.asarray(ref_model.grid.area_t[2:-2, 2:-2])
    dzt = np.asarray(ref_model.grid.dzt)
    ocean3d = np.asarray(ref_model.boundary_conditions.maskT[2:-2, 2:-2, :]) != 0
    vol_weight = area[:, :, None] * dzt[None, None, :] * ocean3d
    vol_denom = vol_weight.sum()
    eq_of_state_type = ref_model.config.eq_of_state_type

    fig, ax = style.plt.subplots(figsize=(7.5, 3.8))
    for n, (c_k, c_eps) in enumerate(common.TOP5):
        d = np.load(common.DATA_DIR / "full_state" / f"ck{c_k:.4g}_eps{c_eps:.4g}.npz")
        temp, salt = d["temp"][:, 2:-2, 2:-2, :], d["salt"][:, 2:-2, 2:-2, :]
        years = (np.arange(temp.shape[0]) + 1) * common.LOG_EVERY_DAYS / 365.0

        rho = get_potential_rho(eq_of_state_type, salt, temp, 0.0)
        step_diff = rho[1:] - rho[:-1]
        rms = np.sqrt((step_diff**2 * vol_weight[None]).sum(axis=(1, 2, 3)) / vol_denom)

        color = style.SERIES[n]
        std = common.rolling_std(rms)
        ax.fill_between(years[1:], np.maximum(rms - std, rms.min() * 0.5), rms + std,
                         color=color, alpha=0.25, lw=0, zorder=0)
        ax.semilogy(years[1:], rms, color=color, lw=1.3, label=f"c_k={c_k:.3g}, c_eps={c_eps:.3g}", zorder=1)

    ax.set_ylabel("RMS(rho[t+1]-rho[t]) (kg/m^3)")
    ax.set_xlabel("model year")
    ax.grid(True, color=style.GRID, lw=0.6)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("step-to-step density change")
    fig.tight_layout()

    out_path = common.FIG_DIR / "fig_density_rms"
    common.FIG_DIR.mkdir(parents=True, exist_ok=True)
    style.save(fig, out_path)


if __name__ == "__main__":
    main()
