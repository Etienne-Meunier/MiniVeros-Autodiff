"""
Reads DATA_DIR/ck_ceps_density_sweep.npz (written by run_sweep.py) and writes:

  report/ck_ceps_density_figures/fig_profiles_vs_ck.{pdf,png}    -- profile lines, c_eps fixed at default, c_k varied
  report/ck_ceps_density_figures/fig_profiles_vs_ceps.{pdf,png}  -- profile lines, c_k fixed at default, c_eps varied
  report/ck_ceps_density_figures/fig_stratification_heatmap.{pdf,png}  -- surface-bottom density contrast over the full (c_k, c_eps) grid
  report/ck_ceps_density_figures/fig_profiles_all_anomaly.{pdf,png}    -- all 25 profiles minus their grid-mean, one line per (c_k, c_eps)

Never re-runs a simulation.
"""

import sys
from pathlib import Path

import numpy as np

import common

sys.path.append(str(Path(__file__).resolve().parents[1] / "paper_figures"))
import style  # noqa: E402

style.use()


def nearest_index(grid, value):
    return int(np.argmin(np.abs(np.asarray(grid) - value)))


def add_cell_index_axis(ax, zt):
    """Right-hand twin y-axis showing the vertical cell index k (0=bottom .. nz-1=surface,
    see model conventions) at each zt depth, alongside the left depth-in-meters axis.
    """
    orange = style.SERIES[1]  # categorical slot 1 is the palette's orange
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(zt)
    ax2.set_yticklabels([str(k) for k in range(len(zt))], fontsize=6, color=orange)
    ax2.set_ylabel("cell index k", color=orange)
    ax2.tick_params(axis="y", colors=orange)
    ax2.spines["right"].set_color(orange)
    ax2.grid(False)
    return ax2


def plot_profiles_vs(k_grid, eps_grid, final_profile, diverged, zt, vary: str, fixed_value: float, out_path: Path):
    """vary='c_k' sweeps k_grid at eps_grid[j0]==fixed_value; vary='c_eps' sweeps eps_grid at k_grid[i0]==fixed_value."""
    fig, ax = style.plt.subplots(figsize=(5.2, 5.6))
    cmap = style.SEQ_BLUE
    if vary == "c_k":
        j0 = nearest_index(eps_grid, fixed_value)
        series, labels = k_grid, [f"c_k={v:.3g}" for v in k_grid]
        title = f"potential density profile vs c_k (c_eps={eps_grid[j0]:.3g})"
    else:
        i0 = nearest_index(k_grid, fixed_value)
        series, labels = eps_grid, [f"c_eps={v:.3g}" for v in eps_grid]
        title = f"potential density profile vs c_eps (c_k={k_grid[i0]:.3g})"

    for n, v in enumerate(series):
        color = cmap(n / max(len(series) - 1, 1))
        if vary == "c_k":
            prof, div = final_profile[n, j0], diverged[n, j0]
        else:
            prof, div = final_profile[i0, n], diverged[i0, n]
        if div or np.all(np.isnan(prof)):
            continue
        ax.plot(prof, zt, color=color, lw=1.8, label=labels[n])

    ax.set_xlabel("potential density anomaly (kg/m^3, rel. rho0=1024, ref. surface)")
    ax.set_ylabel("depth z (m)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, color=style.GRID, lw=0.6)
    add_cell_index_axis(ax, zt)
    style.save(fig, out_path)


def plot_all_anomaly(k_grid, eps_grid, final_profile, diverged, zt, out_path: Path):
    """All 25 (c_k, c_eps) profiles, each minus the grid-mean profile at that depth --
    the mean cancels out the shared channel shape so only each run's deviation from it
    remains. Color encodes c_k (light->dark = low->high), linestyle encodes c_eps.
    """
    ok = ~diverged
    mean_profile = np.nanmean(np.where(ok[:, :, None], final_profile, np.nan), axis=(0, 1))

    fig, ax = style.plt.subplots(figsize=(6.2, 6.4))
    cmap = style.SEQ_BLUE
    linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]  # one per c_eps grid point

    for i, k in enumerate(k_grid):
        color = cmap(i / max(len(k_grid) - 1, 1))
        for j, eps in enumerate(eps_grid):
            if diverged[i, j] or np.all(np.isnan(final_profile[i, j])):
                continue
            anomaly = final_profile[i, j] - mean_profile
            ax.plot(anomaly, zt, color=color, lw=1.4, ls=linestyles[j % len(linestyles)], alpha=0.9)

    ax.axvline(0.0, color=style.INK_MUTED, lw=1.0, zorder=0)
    ax.set_xlabel("potential density anomaly minus grid-mean profile (kg/m^3)")
    ax.set_ylabel("depth z (m)")
    ax.set_title("all 25 (c_k, c_eps) profiles vs their grid-mean")

    ck_handles = [style.plt.Line2D([0], [0], color=cmap(i / max(len(k_grid) - 1, 1)), lw=2.5)
                  for i in range(len(k_grid))]
    ck_labels = [f"c_k={v:.3g}" for v in k_grid]
    leg1 = ax.legend(ck_handles, ck_labels, title="c_k", fontsize=8, loc="upper left")
    ax.add_artist(leg1)

    eps_handles = [style.plt.Line2D([0], [0], color=style.INK_SOFT, lw=1.4, ls=linestyles[j % len(linestyles)])
                   for j in range(len(eps_grid))]
    eps_labels = [f"c_eps={v:.3g}" for v in eps_grid]
    ax.legend(eps_handles, eps_labels, title="c_eps", fontsize=8, loc="lower left")

    ax.grid(True, color=style.GRID, lw=0.6)
    add_cell_index_axis(ax, zt)
    style.save(fig, out_path)


def plot_heatmap(k_grid, eps_grid, final_profile, diverged, out_path: Path):
    """Surface-minus-bottom potential density (a stratification-strength proxy) over the grid."""
    strat = final_profile[:, :, -1] - final_profile[:, :, 0]  # top level is index -1 (see model conventions)
    strat = np.where(diverged, np.nan, strat)

    fig, ax = style.plt.subplots(figsize=(5.6, 5.0))
    finite = strat[np.isfinite(strat)]
    vmin, vmax = (finite.min(), finite.max()) if finite.size else (0, 1)
    mesh = ax.pcolormesh(eps_grid, k_grid, strat, cmap=style.SEQ_BLUE, vmin=vmin, vmax=vmax, shading="nearest")
    for i, k in enumerate(k_grid):
        for j, e in enumerate(eps_grid):
            if diverged[i, j]:
                ax.text(e, k, "x", ha="center", va="center", color=style.INK_SOFT, fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("c_eps")
    ax.set_ylabel("c_k")
    ax.set_title("stratification proxy: rho'(surface) - rho'(bottom)  [kg/m^3 anomaly]")
    style.colorbar(fig, mesh, ax, "kg/m^3")
    style.save(fig, out_path)


def main():
    d = np.load(common.SWEEP_NPZ)
    zt, k_grid, eps_grid = d["zt"], d["k_grid"], d["eps_grid"]
    final_profile, diverged = d["final_profile"], d["diverged"]
    c_k_default, c_eps_default = float(d["c_k_default"]), float(d["c_eps_default"])

    common.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_profiles_vs(k_grid, eps_grid, final_profile, diverged, zt, "c_k", c_eps_default,
                      common.FIG_DIR / "fig_profiles_vs_ck")
    plot_profiles_vs(k_grid, eps_grid, final_profile, diverged, zt, "c_eps", c_k_default,
                      common.FIG_DIR / "fig_profiles_vs_ceps")
    plot_heatmap(k_grid, eps_grid, final_profile, diverged, common.FIG_DIR / "fig_stratification_heatmap")
    plot_all_anomaly(k_grid, eps_grid, final_profile, diverged, zt, common.FIG_DIR / "fig_profiles_all_anomaly")
    n_div = int(diverged.sum())
    print(f"wrote figures to {common.FIG_DIR} ({n_div}/{diverged.size} runs diverged)")


if __name__ == "__main__":
    main()
