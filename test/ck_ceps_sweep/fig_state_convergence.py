"""
Reads the 5 full-state runs written by run_top5_full_state.py and writes
report/ck_ceps_density_figures/fig_state_convergence.{pdf,png}: 8 panels
(u, v, temp, salt, tke, eke, psi, and a combined "total") stacked, each
showing that field's step-to-step change RMS(field[t+1]-field[t])
(volume/area-weighted, ocean cells only), normalized by the field's own
whole-run RMS magnitude so all 7 fields land on a comparable (0, ~1) scale
despite different units -- this is fig_convergence_metrics.py's density-only
convergence check (panel 3 there), extended to the whole prognostic state.
The "total" panel combines all 7 into one number per run:
sqrt(mean over fields of normalized_step_rms^2).

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

FIELDS = ["u", "v", "temp", "salt", "tke", "eke", "psi"]


def weighted_rms(field, weight):
    """RMS of `field` over its trailing spatial axes and its leading (time) axis together."""
    axes = tuple(range(1, field.ndim))
    per_t = (field**2 * weight).sum(axis=axes) / weight.sum()
    return np.sqrt(per_t.mean())


def main():
    ref_model, _, _ = full.build({})
    area = np.asarray(ref_model.grid.area_t[2:-2, 2:-2])
    dzt = np.asarray(ref_model.grid.dzt)
    ocean3d = np.asarray(ref_model.boundary_conditions.maskT[2:-2, 2:-2, :]) != 0
    vol_weight = area[:, :, None] * dzt[None, None, :] * ocean3d
    area_weight = area * ocean3d[:, :, -1]

    normalized_by_field = {name: [] for name in FIELDS}
    years_rms = None

    for c_k, c_eps in common.TOP5:
        path = common.full_state_path(c_k, c_eps)
        d = np.load(path)
        n_logs = d["temp"].shape[0]
        years = (np.arange(n_logs) + 1) * common.LOG_EVERY_DAYS / 365.0
        years_rms = years[1:]

        for name in FIELDS:
            field = d[name][:, 2:-2, 2:-2, ...]
            weight = area_weight if name == "psi" else vol_weight
            scale = weighted_rms(field, weight)
            step_diff = field[1:] - field[:-1]
            axes = tuple(range(1, field.ndim))
            step_rms = np.sqrt((step_diff**2 * weight).sum(axis=axes) / weight.sum())
            normalized_by_field[name].append(step_rms / max(scale, 1e-30))

    total_by_run = [
        np.sqrt(np.mean([normalized_by_field[name][n] ** 2 for name in FIELDS], axis=0))
        for n in range(len(common.TOP5))
    ]

    fig, axes = style.plt.subplots(len(FIELDS) + 1, 1, figsize=(7.5, 2.0 * (len(FIELDS) + 1)), sharex=True)
    for ax, name in zip(axes, FIELDS):
        for n, (c_k, c_eps) in enumerate(common.TOP5):
            color = style.SERIES[n]
            series = normalized_by_field[name][n]
            std = common.rolling_std(series)
            ax.fill_between(years_rms, series - std, series + std, color=color, alpha=0.2, lw=0, zorder=0)
            label = f"c_k={c_k:.3g}, c_eps={c_eps:.3g}" if name == FIELDS[0] else None
            ax.semilogy(years_rms, series, color=color, lw=1.2, label=label, zorder=1)
        ax.set_ylabel(name)
        ax.grid(True, color=style.GRID, lw=0.6)

    ax = axes[-1]
    for n, (c_k, c_eps) in enumerate(common.TOP5):
        color = style.SERIES[n]
        series = total_by_run[n]
        std = common.rolling_std(series)
        ax.fill_between(years_rms, series - std, series + std, color=color, alpha=0.2, lw=0, zorder=0)
        ax.semilogy(years_rms, series, color=color, lw=1.4, zorder=1)
    ax.set_ylabel("total")
    ax.grid(True, color=style.GRID, lw=0.6)

    axes[-1].set_xlabel("model year")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("normalized step-to-step change per field, and combined ('total'),\n"
                 "RMS(field[t+1]-field[t]) / whole-run RMS(field) -- log scale", y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    out_path = common.FIG_DIR / "fig_state_convergence"
    common.FIG_DIR.mkdir(parents=True, exist_ok=True)
    style.save(fig, out_path)


if __name__ == "__main__":
    main()
