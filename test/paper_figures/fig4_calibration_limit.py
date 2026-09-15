"""
Figure 4 -- the calibration limit.

(a) how far the fitted parameters end up from the truth, per starting point;
(b) how much the optimiser reduced the loss it was actually given -- the two
    disagree past a point, which is the failure worth naming: the fit succeeds
    and the parameters are still wrong;
(c) the fitted values themselves, normalised by the truth, so the drift is
    readable parameter by parameter.

    python test/paper_figures/fig4_calibration_limit.py
"""

import matplotlib.pyplot as plt
import numpy as np

import common
import style

YEAR = 365 * 86400


def main():
    style.use()
    d = common.load("exp4_calibration_limit")
    n, names, truth = d["lengths"], [str(p) for p in d["param_names"]], d["true"]
    fitted, dt = d["fitted"], float(d["dt_tracer"])  # (length, start, param)
    horizon_days = int(common.load("exp1_gradient_check")["horizon_days"])

    rel_err = np.sqrt(np.mean(((fitted - truth) / truth) ** 2, axis=2))  # (length, start)
    reduction = d["loss_end"] / d["loss_start"]
    ratio = fitted / truth

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    for j in range(rel_err.shape[1]):
        label = "start " + ", ".join(f"{p}={v:.2f}" for p, v in zip(names, d["starts"][j]))
        axes[0].plot(n, rel_err[:, j], color=style.SERIES[j], marker="o", label=label)
        axes[1].plot(n, reduction[:, j], color=style.SERIES[j], marker="o", label=label)

    axes[0].axhline(1.0, color=style.INK_MUTED, lw=0.9, ls=(0, (4, 3)))
    axes[0].annotate("as wrong as guessing", (n[0], 1.0), textcoords="offset points", xytext=(2, 4),
                     fontsize=8, color=style.INK_MUTED)
    axes[0].set_title("(a) distance to the true parameters")
    axes[0].set_ylabel(r"rms$\,[(\hat p - p^\star)/p^\star]$")
    axes[0].legend(loc="lower right")

    axes[1].set_title("(b) loss the optimiser reached")
    axes[1].set_ylabel(r"$L_{\rm final}\,/\,L_{\rm initial}$")

    for k, name in enumerate(names):
        c = style.SERIES[k]
        axes[2].fill_between(n, ratio[:, :, k].min(axis=1), ratio[:, :, k].max(axis=1),
                             color=c, alpha=0.15, lw=0)
        axes[2].plot(n, np.median(ratio[:, :, k], axis=1), color=c, marker="o", label=name)
    axes[2].axhline(1.0, color=style.INK_MUTED, lw=0.9, ls=(0, (4, 3)))
    axes[2].set_yscale("linear")
    axes[2].set_title(f"(c) fitted / true, {', '.join(names)}")
    axes[2].set_ylabel(r"$\hat p\,/\,p^\star$")
    if len(names) > 1:  # a single series is named by the title; no legend box needed
        axes[2].legend(loc="upper left", ncols=2)

    # Where figure 1 puts the horizon H: autodiff still matches finite differences
    # inside it and disagrees past it.
    days_per_year = YEAR / 86400
    for ax in axes:
        ax.axvline(horizon_days, color=style.INK_MUTED, lw=0.9, ls=(0, (1, 2)), zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel(f"rollout length (days; {days_per_year:.0f} d = 1 model year)")
        ax.set_xlim(n.min() * 0.85, n.max() * 1.2)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[1].annotate(f"H = {horizon_days} d\n(figure 1)", (horizon_days, 0.9), xycoords=("data", "axes fraction"),
                     ha="center", va="top", fontsize=8, color=style.INK_MUTED)

    fig.suptitle(f"Fitting {', '.join(names)} over rollouts of growing length",
                 fontsize=10.5, y=1.04, color=style.INK)
    fig.tight_layout()
    style.save(fig, common.FIG_DIR / "fig4_calibration_limit")


if __name__ == "__main__":
    main()
