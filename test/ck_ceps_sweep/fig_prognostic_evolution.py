"""
Reads the 5 full-state runs written by run_top5_full_state.py
(DATA_DIR/full_state/ck{c_k}_eps{c_eps}.npz) and writes
report/ck_ceps_density_figures/fig_prognostic_evolution.{pdf,png}: one panel
per prognostic field (u, v, temp, salt, tke, eke, psi), stacked vertically,
each panel showing that field's basin-average through the full 30-year run
for all 5 runs overlaid, each run's own colour also shading a +/-1 local
rolling-std band around its own line (common.rolling_std: how much that
run's series wiggles near each point in time, not its spatial spread within
the basin -- the latter is ~10-100x the basin mean for a field like u or
psi, whose basin average is a near-cancellation of large positive and
negative regions, and plotting it swamps the panel).

Basin average: volume-weighted (area_t x dzt), ocean cells only, for the 3D
fields; area-weighted (area_t), ocean surface cells only, for psi (2D).

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

FIELDS_3D = ["u", "v", "temp", "salt", "tke", "eke"]
UNITS = {"u": "m/s", "v": "m/s", "temp": "deg C", "salt": "psu",
         "tke": "m^2/s^2", "eke": "m^2/s^2", "psi": "model units"}


def main():
    ref_model, _, _ = full.build({})
    area = np.asarray(ref_model.grid.area_t[2:-2, 2:-2])              # (nx, ny)
    dzt = np.asarray(ref_model.grid.dzt)                              # (nz,)
    ocean3d = np.asarray(ref_model.boundary_conditions.maskT[2:-2, 2:-2, :]) != 0  # (nx, ny, nz)
    vol_weight = area[:, :, None] * dzt[None, None, :] * ocean3d
    vol_denom = vol_weight.sum()
    area_weight = area * ocean3d[:, :, -1]
    area_denom = area_weight.sum()

    fields = FIELDS_3D + ["psi"]
    series_by_field = {name: [] for name in fields}  # name -> list of 5 (n_logs,) arrays
    years = None

    for c_k, c_eps in common.TOP5:
        path = common.full_state_path(c_k, c_eps)
        d = np.load(path)
        n_logs = d["temp"].shape[0]
        years = (np.arange(n_logs) + 1) * common.LOG_EVERY_DAYS / 365.0
        for name in fields:
            field = d[name][:, 2:-2, 2:-2, ...]  # (n_logs, nx, ny[, nz])
            weight = area_weight if name == "psi" else vol_weight
            mean, _ = common.weighted_stats(field, weight)
            series_by_field[name].append(mean)

    fig, axes = style.plt.subplots(len(fields), 1, figsize=(7.5, 2.0 * len(fields)), sharex=True)
    for ax, name in zip(axes, fields):
        for n, (c_k, c_eps) in enumerate(common.TOP5):
            series = series_by_field[name][n]
            std = common.rolling_std(series)
            color = style.SERIES[n]
            ax.fill_between(years, series - std, series + std, color=color, alpha=0.2, lw=0, zorder=0)
            label = f"c_k={c_k:.3g}, c_eps={c_eps:.3g}" if name == fields[0] else None
            ax.plot(years, series, color=color, lw=1.3, label=label, zorder=1)
        ax.set_ylabel(f"{name} ({UNITS[name]})")
        ax.grid(True, color=style.GRID, lw=0.6)
    axes[-1].set_xlabel("model year")
    axes[0].legend(fontsize=8, loc="upper right", ncol=1)
    fig.suptitle("basin-averaged prognostic fields through the 30-year run", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = common.FIG_DIR / "fig_prognostic_evolution"
    common.FIG_DIR.mkdir(parents=True, exist_ok=True)
    style.save(fig, out_path)


if __name__ == "__main__":
    main()
