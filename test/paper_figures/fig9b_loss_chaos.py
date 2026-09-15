"""
Figure 9b -- averaging quiets the finite difference, but the horizon doesn't move.

Two panels, one line per loss (final_mse, and the two time-averaged and one
space-averaged variant), against rollout length:
  (a) the finite difference's own round-off-ensemble spread (relative std
      across members) -- the prediction is that this shrinks for the
      averaged losses relative to final_mse.
  (b) reverse-mode-vs-finite-difference relative error (member 0, the
      unperturbed state) -- the prediction is that these curves stay on top
      of each other regardless of averaging: the adjoint walks the same
      unaveraged dynamics either way.

    python test/paper_figures/fig9b_loss_chaos.py
"""

import matplotlib.pyplot as plt
import numpy as np

import common
import style

LOSS_LABELS = {
    "final_mse": "final step (no averaging)",
    "avg_mse_full": "time-averaged, full rollout",
    "avg_mse_last30": "time-averaged, last 30 d",
    "global_mean_sst_sq": "space-averaged (mean before square)",
}


def main():
    style.use()
    d = common.load("exp9b_loss_chaos")
    n = d["lengths"]
    loss_names = [str(name) for name in d["loss_names"]]
    grad_ad, grad_fd = d["grad_ad"], d["grad_fd"]  # (length, member, loss)

    spread = np.nanstd(grad_fd, axis=1) / np.maximum(np.abs(np.nanmedian(grad_fd, axis=1)), 1e-30)  # (length, loss)
    rel_err = np.abs(grad_ad[:, 0] - grad_fd[:, 0]) / np.abs(grad_fd[:, 0])  # (length, loss)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    for j, name in enumerate(loss_names):
        ax_a.plot(n, spread[:, j], color=style.SERIES[j], marker="o", ms=4, label=LOSS_LABELS.get(name, name))
        ax_b.plot(n, rel_err[:, j], color=style.SERIES[j], marker="o", ms=4, label=LOSS_LABELS.get(name, name))

    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rollout length (days)")

    ax_a.set_ylabel("relative std of finite difference across round-off ensemble")
    ax_a.set_title("(a) does averaging quiet the finite difference?")
    ax_a.legend(fontsize=7.5)

    ax_b.set_ylabel(r"$|g_{\rm AD} - g_{\rm FD}|\ /\ |g_{\rm FD}|$  (member 0)")
    ax_b.set_title("(b) does averaging move the AD-vs-FD horizon?")

    fig.suptitle("Averaging smooths the finite difference's own noise, not the adjoint's disagreement with it",
                 fontsize=10.5, color=style.INK, y=1.08)
    style.save(fig, common.FIG_DIR / "fig9b_loss_chaos")


if __name__ == "__main__":
    main()
