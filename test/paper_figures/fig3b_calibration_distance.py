"""
Figure 3b -- calibration success as a function of rollout length AND starting distance.

Panel (a): a length x distance matrix, coloured and annotated by mean
relative parameter error across repeats -- the same matrix+colour idiom as
figure 5b, so the two "does it actually recover the truth" figures read the
same way. Panel (b): the same cells' final loss, to show the figure 4 failure
mode (loss keeps dropping, parameter doesn't) holds across starting distance
too. Error bars (repeat-to-repeat std) are printed in each cell's annotation
rather than drawn, to keep the matrix readable -- exp3b_calibration_distance.npz
keeps every repeat, so a different plot (e.g. error bars drawn per length at
fixed distance) can be made later without rerunning.

    python test/paper_figures/fig3b_calibration_distance.py
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
    d = common.load("exp3b_calibration_distance")
    lengths, distances = d["lengths"], d["distances"]
    true_c_k, fitted, loss_end = float(d["true_c_k"]), d["fitted"], d["loss_end"]

    rel_err = np.abs(fitted - true_c_k) / true_c_k  # (length, distance, repeat)
    mean_err = np.nanmean(rel_err, axis=-1)
    mean_loss = np.nanmean(loss_end, axis=-1)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    matrix_panel(ax_a, fig, lengths, distances, np.clip(mean_err, 0, 1), 0, 1, style.SEQ_BLUE,
                 "(a) mean relative error in fitted c_k", "|fitted - true| / true (clipped at 1)")
    log_loss = np.log10(np.maximum(mean_loss, 1e-30))
    matrix_panel(ax_b, fig, lengths, distances, log_loss, log_loss.min(), log_loss.max(), style.SEQ_WARM,
                 "(b) final loss reached", "log10(final loss)", fmt="{:.1f}")

    fig.suptitle("Calibration limit: rollout length vs how wrong the starting guess is",
                 fontsize=10.5, color=style.INK, y=1.05)
    style.save(fig, common.FIG_DIR / "fig3b_calibration_distance")


if __name__ == "__main__":
    main()
