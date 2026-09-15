"""
Figure 5b -- the assimilation success/failure map: metric x rollout length.

Panel (a) is the point: for every (rollout length, observation metric) pair,
how much of the initial-state error is left after the fit, as a fraction of
where it started. Near 0 = recovered; near 1 = the fit did nothing to the
whole-volume state (even though, as in figure 5a, the *observed* loss can
still have dropped hard). Panels (b) and (c) open up two slices of that map
into rms error by depth, to show *where* the recovery stops rather than just
by how much: (b) fixes the metric figure 5 used (T,S on the top 3 layers)
and varies length, (c) fixes the length at the horizon H and varies the
metric.

    python test/paper_figures/fig5b_assimilation_horizon.py
"""

import matplotlib.pyplot as plt
import numpy as np

import common
import style

METRIC_LABELS = {
    "sst_mse": "SST only (MSE)",
    "top1_TS": "top 1 layer, T+S",
    "top3_TS": "top 3 layers, T+S",
    "top6_TS": "top 6 layers, T+S",
    "full_TS": "all layers, T+S",
}


def ocean_rms_by_depth(field, land):
    """rms over horizontal ocean points, one value per depth level. field: (nx, ny, nz)."""
    ocean = ~land
    return np.sqrt(np.array([np.mean(field[..., k][ocean] ** 2) for k in range(field.shape[-1])]))


def main():
    style.use()
    d = common.load("exp5b_assimilation_horizon")
    lengths, metrics, land, z = d["lengths"], list(d["metrics"]), d["land"], d["z"]
    n_layers, truth, fitted = d["n_layers"], d["truth"], d["fitted"]
    state_err_start, state_err_end = d["state_err_start"], d["state_err_end"]
    recovery = state_err_end / state_err_start  # (len(lengths), len(metrics))

    fig = plt.figure(figsize=(12.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    ax_map, ax_len, ax_metric = (fig.add_subplot(gs[0, i]) for i in range(3))

    mesh = ax_map.pcolormesh(np.arange(len(lengths) + 1), np.arange(len(metrics) + 1), recovery.T,
                              cmap=style.SEQ_BLUE, vmin=0, vmax=1, rasterized=True)
    for i in range(len(lengths)):
        for j in range(len(metrics)):
            ax_map.annotate(f"{recovery[i, j]:.2f}", (i + 0.5, j + 0.5), ha="center", va="center",
                             fontsize=8, color=style.INK if recovery[i, j] > 0.55 else style.SURFACE)
    ax_map.set_xticks(np.arange(len(lengths)) + 0.5)
    ax_map.set_xticklabels(lengths)
    ax_map.set_yticks(np.arange(len(metrics)) + 0.5)
    ax_map.set_yticklabels([METRIC_LABELS.get(m, m) for m in metrics])
    ax_map.set_xlabel("rollout length (days)")
    ax_map.set_title("(a) fraction of state error left after the fit")
    ax_map.grid(False)
    style.colorbar(fig, mesh, ax_map, "state err(end) / state err(start)")

    def depth_curve(ax, i, j, color, label):
        err = ocean_rms_by_depth(fitted[i, j] - truth, land)
        ax.plot(err, z, color=color, marker="o", ms=3, label=label)

    ref_metric = "top3_TS" if "top3_TS" in metrics else metrics[len(metrics) // 2]
    j_ref = metrics.index(ref_metric)
    for i, n in enumerate(lengths):
        depth_curve(ax_len, i, j_ref, style.SERIES[i % len(style.SERIES)], f"{n} d")
    ax_len.set_xlabel("rms recovered $-$ truth (K)")
    ax_len.set_ylabel("depth (m)")
    ax_len.set_title(f"(b) fixed metric ({METRIC_LABELS.get(ref_metric, ref_metric)}), by length")
    ax_len.legend(fontsize=7.5, ncols=2, title="rollout")

    ref_len = 80 if 80 in list(lengths) else lengths[len(lengths) // 2]
    i_ref = list(lengths).index(ref_len)
    for j, m in enumerate(metrics):
        depth_curve(ax_metric, i_ref, j, style.SERIES[j % len(style.SERIES)], METRIC_LABELS.get(m, m))
    ax_metric.set_xlabel("rms recovered $-$ truth (K)")
    ax_metric.set_title(f"(c) fixed length ({ref_len} d = H), by metric")
    ax_metric.legend(fontsize=7.5)

    fig.suptitle("Assimilating a full-depth initial temperature perturbation: "
                  "when the fit actually recovers it, and when it doesn't", fontsize=11, color=style.INK, y=1.06)
    style.save(fig, common.FIG_DIR / "fig5b_assimilation_horizon")


if __name__ == "__main__":
    main()
