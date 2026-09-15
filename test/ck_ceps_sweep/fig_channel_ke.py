"""
Reads the 5 full-state runs and writes report/ck_ceps_density_figures/
fig_channel_ke.{pdf,png}: mean(u^2+v^2) in the periodic part of the channel
(y<-20, south of the "Atlantic block" landmass -- the ACC proper),
volume-weighted, through the 30-year run, all 5 runs overlaid (each with its
own colour shading a local rolling-std band).

Never re-runs a simulation.
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
    ref_model, _, _ = full.build({})
    y = np.asarray(ref_model.grid.yt[2:-2])
    area = np.asarray(ref_model.grid.area_t[2:-2, 2:-2])
    dzt = np.asarray(ref_model.grid.dzt)
    ocean3d = np.asarray(ref_model.boundary_conditions.maskT[2:-2, 2:-2, :]) != 0
    channel_weight = area[:, :, None] * dzt[None, None, :] * ocean3d * (y < -20)[None, :, None]
    channel_denom = channel_weight.sum()

    fig, ax = style.plt.subplots(figsize=(7.5, 3.8))
    for n, (c_k, c_eps) in enumerate(common.TOP5):
        d = np.load(common.DATA_DIR / "full_state" / f"ck{c_k:.4g}_eps{c_eps:.4g}.npz")
        u, v = d["u"][:, 2:-2, 2:-2, :], d["v"][:, 2:-2, 2:-2, :]
        years = (np.arange(u.shape[0]) + 1) * common.LOG_EVERY_DAYS / 365.0
        ke = ((u**2 + v**2) * channel_weight[None]).sum(axis=(1, 2, 3)) / channel_denom

        color = style.SERIES[n]
        std = common.rolling_std(ke)
        ax.fill_between(years, ke - std, ke + std, color=color, alpha=0.25, lw=0, zorder=0)
        ax.plot(years, ke, color=color, lw=1.3, label=f"c_k={c_k:.3g}, c_eps={c_eps:.3g}", zorder=1)

    ax.set_ylabel("<u^2+v^2> in y<-20 (m^2/s^2)")
    ax.set_xlabel("model year")
    ax.grid(True, color=style.GRID, lw=0.6)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("kinetic energy in the periodic part of the channel")
    fig.tight_layout()

    out_path = common.FIG_DIR / "fig_channel_ke"
    common.FIG_DIR.mkdir(parents=True, exist_ok=True)
    style.save(fig, out_path)


if __name__ == "__main__":
    main()
