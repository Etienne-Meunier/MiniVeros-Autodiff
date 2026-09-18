"""
Figure 11 -- how the (c_k, c_eps) loss landscape changes with rollout length.

One panel per length in exp11's sweep, shortest to longest: log10 loss landscape, with the
3 starting directions' descent paths for one distance (`--distance-index`, default the
middle one) drawn on top. The two failure modes should show up as landscape shape: a ridge
rather than a bowl at the shortest length (parameters barely identifiable), and paths that
keep moving downhill without reaching the truth at the longest (figure 11b puts a number on
both).

    python test/paper_figures/fig11_density_calibration_limit_landscapes.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import common
import style


def main(distance_index):
    style.use()
    d = common.load("exp11_density_calibration_limit")
    lengths, distances = d["lengths"], d["distances"]
    true, names = d["true"], [str(p) for p in d["param_names"]]
    landscape, trajectories = d["landscape"], d["trajectories"]
    if distance_index is None:
        distance_index = len(distances) // 2
    dist = float(distances[distance_index])

    n = len(lengths)
    fig, axes = plt.subplots(1, n, figsize=(2.85 * n, 3.4), constrained_layout=True, sharey=True)
    for i, ax in enumerate(axes):
        floor = np.nanmin(landscape[i][landscape[i] > 0]) / 2
        logL = np.log10(np.maximum(landscape[i], floor))
        mesh = ax.contourf(d["c_k"], d["c_eps"], logL.T, levels=20, cmap="viridis")
        ax.contour(d["c_k"], d["c_eps"], logL.T, levels=10, colors=style.SURFACE, linewidths=0.4, alpha=0.5)

        # The 3 starting directions sit at one distance from the truth in (log c_k, log c_eps)
        # space -- its isoline there is a circle, which maps to this curve in linear (c_k, c_eps).
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(true[0] * np.exp(dist * np.cos(theta)), true[1] * np.exp(dist * np.sin(theta)),
                color='red', lw=0.9, ls=(0, (3, 2)), zorder=2, alpha=0.4)

        for a in range(trajectories.shape[2]):
            traj = trajectories[i, distance_index, a]
            color = style.SERIES[a]
            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.4,
                    marker="o", ms=2.5, markevery=max(1, len(traj) // 10), mec=style.SURFACE, mew=0.3, zorder=3)
            ax.plot(*traj[0], marker="o", ms=7, mfc="none", mec=color, mew=1.5, ls="none", zorder=4)
        ax.plot(*true, marker="*", ms=13, color=style.INK, mec=style.SURFACE, mew=0.5, ls="none", zorder=5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(d["c_k"].min(), d["c_k"].max())
        ax.set_ylim(d["c_eps"].min(), d["c_eps"].max())
        style.natural_log_ticks(ax, [0.05, 0.10, 0.15, 0.25], [0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="both", labelsize=8)
        ax.set_title(f"{int(lengths[i])} d", fontsize=9.5)
        ax.set_xlabel(names[0])
        ax.grid(False)
        ax.tick_params(axis="y", labelleft=(i == 0))
        if i == 0:
            ax.set_ylabel(names[1])
    style.colorbar(fig, mesh, list(axes), r"$\log_{10}$ loss")

    fig.suptitle(f"Loss landscape vs rollout length · start distance |log(start/true)| = {dist:.2f} "
                 f"(3 directions)", fontsize=10.5, color=style.INK, y=1.08)
    style.save(fig, common.FIG_DIR / "fig11_density_calibration_limit_landscapes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-index", type=int, default=None,
                        help="which of exp11's DISTANCES to draw paths for [default: the middle one]")
    args = parser.parse_args()
    main(args.distance_index)
