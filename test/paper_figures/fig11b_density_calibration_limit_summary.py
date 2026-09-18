"""
Figure 11b -- calibration limit summary: rollout length vs starting distance.

Panel (a): a length x distance matrix, coloured and annotated by mean relative parameter
error (rms over c_k, c_eps, averaged over the 3 starting directions) -- the same matrix+colour
idiom as figure 3b/5b. Panel (b): the same cells' final loss, to show the failure mode
(loss keeps dropping, parameters don't) alongside the ridge/bowl story figure 11 draws for the
same cells. Direction-to-direction spread is printed in each cell's annotation rather than
drawn, to keep the matrix readable -- exp11_density_calibration_limit.npz keeps every
direction's trajectory, so a different summary can be made later without rerunning.

    python test/paper_figures/fig11b_density_calibration_limit_summary.py
"""

import matplotlib.pyplot as plt
import numpy as np

import common
import style


def matrix_panel(ax, fig, lengths, distances, values, vmin, vmax, cmap, title, cbar_label, fmt="{:.2f}"):
    mesh = ax.pcolormesh(np.arange(len(lengths) + 1), np.arange(len(distances) + 1), values.T,
                          cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    mid = vmin + 0.55 * (vmax - vmin)
    for i in range(len(lengths)):
        for j in range(len(distances)):
            color = style.INK if values[i, j] < mid else style.SURFACE
            ax.annotate(fmt.format(values[i, j]), (i + 0.5, j + 0.5), ha="center", va="center",
                        fontsize=7.5, color=color)
    ax.set_xticks(np.arange(len(lengths)) + 0.5)
    ax.set_xticklabels(lengths)
    ax.set_yticks(np.arange(len(distances)) + 0.5)
    ax.set_yticklabels([f"{d:.2f}" for d in distances])
    ax.set_xlabel("rollout length (days)")
    ax.set_ylabel("start distance |log(start/true)|")
    ax.set_title(title)
    ax.grid(False)
    style.colorbar(fig, mesh, ax, cbar_label)


def main():
    style.use()
    d = common.load("exp11_density_calibration_limit")
    lengths, distances = d["lengths"], d["distances"]
    true, fitted, loss_end = d["true"], d["fitted"], d["loss_end"]  # fitted: (length, distance, direction, param)

    rel_err = np.sqrt(np.mean(((fitted - true) / true) ** 2, axis=-1))  # (length, distance, direction)
    mean_err = np.nanmean(rel_err, axis=-1)
    mean_loss = np.nanmean(loss_end, axis=-1)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    matrix_panel(ax_a, fig, lengths, distances, np.clip(mean_err, 0, 1), 0, 1, style.SEQ_BLUE,
                 "(a) mean relative error in fitted (c_k, c_eps)", "rms[(fitted - true) / true] (clipped at 1)")
    log_loss = np.log10(np.maximum(mean_loss, 1e-30))
    matrix_panel(ax_b, fig, lengths, distances, log_loss, np.nanmin(log_loss), np.nanmax(log_loss), style.SEQ_WARM,
                 "(b) final loss reached", "log10(final loss)", fmt="{:.1f}")

    fig.suptitle("Density-gradient calibration limit: rollout length vs starting distance",
                 fontsize=10.5, color=style.INK, y=1.05)
    style.save(fig, common.FIG_DIR / "fig11b_density_calibration_limit_summary")


if __name__ == "__main__":
    main()
