"""
Figure 3c -- joint (c_k, c_eps) calibration: rollout length x how far off the start is (2D),
and whether the direction of that initial error matters.

Panel (a): a length x distance matrix, coloured and annotated by the recovered point's own
log-space distance from truth, averaged over the 3 angles tested at that (length, distance) --
same units as the sampling radius itself, so 0 means "landed back on truth" and the untouched
radius (e.g. 1.0) means "didn't move at all". Panels (b)/(c) open the angle dimension back up:
(b) fixes the largest distance and shows recovery vs length, one line per angle; (c) fixes the
length at the horizon H and shows recovery vs distance, one line per angle. If the three lines
in (b)/(c) sit on top of each other, the direction of the initial error doesn't matter, only
its size; if they fan out, it does.

    python test/paper_figures/fig3c_calibration_joint.py
"""

import matplotlib.pyplot as plt
import numpy as np

import common
import style


def recovered_distance(fitted, true):
    """log-space L2 distance of `fitted` (..., 2) from `true` (2,) -- same metric the sampling
    radius itself is defined in."""
    log_ratio = np.log(fitted / true)
    return np.sqrt((log_ratio ** 2).sum(axis=-1))


def main():
    style.use()
    d = common.load("exp3c_calibration_joint")
    lengths, distances, angles = d["lengths"], d["distances"], d["angles_deg"]
    true, fitted = d["true"], d["fitted"]  # fitted: (length, distance, angle, 2)

    recovery = recovered_distance(fitted, true)  # (length, distance, angle)
    mean_recovery = recovery.mean(axis=-1)

    fig, (ax_map, ax_len, ax_dist) = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)

    vmax = float(np.nanmax(distances))
    mesh = ax_map.pcolormesh(np.arange(len(lengths) + 1), np.arange(len(distances) + 1), mean_recovery.T,
                              cmap=style.SEQ_BLUE, vmin=0, vmax=vmax, rasterized=True)
    for i in range(len(lengths)):
        for j in range(len(distances)):
            ax_map.annotate(f"{mean_recovery[i, j]:.2f}", (i + 0.5, j + 0.5), ha="center", va="center",
                             fontsize=7.5, color=style.INK if mean_recovery[i, j] < 0.55 * vmax else style.SURFACE)
    ax_map.set_xticks(np.arange(len(lengths)) + 0.5)
    ax_map.set_xticklabels(lengths)
    ax_map.set_yticks(np.arange(len(distances)) + 0.5)
    ax_map.set_yticklabels([f"{dd:.2f}" for dd in distances])
    ax_map.set_xlabel("rollout length (days)")
    ax_map.set_ylabel("start distance (log-space L2 radius)")
    ax_map.set_title("(a) recovered distance from truth (mean over angles)")
    ax_map.grid(False)
    style.colorbar(fig, mesh, ax_map, "log-space L2 distance of fit from truth")

    j_far = len(distances) - 1
    for k, angle in enumerate(angles):
        ax_len.plot(lengths, recovery[:, j_far, k], color=style.SERIES[k], marker="o", ms=4,
                    label=f"{angle:.0f}°")
    ax_len.axhline(distances[j_far], color=style.GRID, lw=0.9, ls=(0, (1, 2)))
    ax_len.set_xscale("log")
    ax_len.set_xlabel("rollout length (days)")
    ax_len.set_ylabel("recovered distance from truth")
    ax_len.set_title(f"(b) fixed distance ({distances[j_far]:.2f}), by angle")
    ax_len.legend(fontsize=8, title="start angle")

    i_ref = int(np.argmin(np.abs(np.array(lengths) - 80)))
    for k, angle in enumerate(angles):
        ax_dist.plot(distances, recovery[i_ref, :, k], color=style.SERIES[k], marker="o", ms=4,
                     label=f"{angle:.0f}°")
    ax_dist.plot(distances, distances, color=style.INK_MUTED, lw=0.9, ls=(0, (1, 2)), label="no recovery")
    ax_dist.set_xlabel("start distance (log-space L2 radius)")
    ax_dist.set_title(f"(c) fixed length ({lengths[i_ref]} d), by angle")
    ax_dist.legend(fontsize=8, title="start angle")

    fig.suptitle("Joint (c_k, c_eps) calibration: does the direction of the initial error matter, "
                  "or only its size?", fontsize=10.5, color=style.INK, y=1.08)
    style.save(fig, common.FIG_DIR / "fig3c_calibration_joint")


if __name__ == "__main__":
    main()
