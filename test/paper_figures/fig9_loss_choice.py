"""
Figure 9 -- does the loss choice buy horizon?

(a) relative disagreement between autodiff and finite differences against
    rollout length, one line per loss (figure 1's panel (a), repeated for six
    losses instead of three parameters).
(b) each loss's horizon H -- the longest length inside 1% -- as a bar chart,
    so the six can be compared at a glance.

    python test/paper_figures/fig9_loss_choice.py
"""

import matplotlib.pyplot as plt
import numpy as np

import common
import style


def main():
    style.use()
    d = common.load("exp9_loss_choice")
    n, names = d["lengths"], [str(l) for l in d["loss_names"]]
    ad, fd, horizon = d["grad_ad"], d["grad_fd"], d["horizon"]

    rel_err = np.abs(ad - fd) / np.abs(fd)  # (length, loss)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw={"width_ratios": [1.6, 1.0]})

    ax = axes[0]
    for j, name in enumerate(names):
        ax.plot(n, rel_err[:, j], color=style.SERIES[j], label=name)
    ax.axhline(1.0, color=style.INK_MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.annotate("no digits left", (n[0], 1.0), textcoords="offset points", xytext=(2, 5),
               fontsize=8, color=style.INK_MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rollout length (days)")
    ax.set_ylabel(r"$|g_{\rm AD} - g_{\rm FD}|\ /\ |g_{\rm FD}|$")
    ax.set_title("(a) autodiff vs finite difference, by loss")
    style.label_series_ends(ax, n, [(rel_err[:, j], name, style.SERIES[j]) for j, name in enumerate(names)])

    ax = axes[1]
    ax.bar(range(len(names)), horizon, color=style.SERIES[: len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("horizon H (days)")
    ax.set_title("(b) horizon per loss")

    fig.suptitle(f"Does the loss choice buy horizon? (w.r.t. {str(d['param'])})",
                fontsize=10.5, y=1.03, color=style.INK)
    fig.tight_layout()
    style.save(fig, common.FIG_DIR / "fig9_loss_choice")


if __name__ == "__main__":
    main()
