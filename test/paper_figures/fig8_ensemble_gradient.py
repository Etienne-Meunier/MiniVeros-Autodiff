"""
Figure 8 -- can an ensemble average the blow-up away?

(a) every member's gradient, autodiff against finite difference, one column per
    rollout length. Inside the horizon the two clouds sit on top of each other;
    past it the finite differences stay put while the autodiff gradients spread
    over decades.
(b) the running ensemble mean of the autodiff gradient, as members are added,
    divided by the ensemble-mean finite difference. Converging to 1 would mean
    the blow-up is noise and an ensemble is the fix. A curve that wanders
    without settling means it is not.
(c) the two ensemble means side by side, per length.

    python test/paper_figures/fig8_ensemble_gradient.py
"""

import matplotlib.pyplot as plt
import numpy as np

import common
import style


def main():
    style.use()
    d = common.load("exp8_ensemble_gradient")
    lengths, param = d["lengths"], str(d["param"])
    ad, fd = d["grad_ad"], d["grad_fd"]  # (length, member)
    members = np.arange(1, ad.shape[1] + 1)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    # (a) per-member clouds, jittered so overlapping points stay countable.
    rng = np.random.default_rng(0)
    for i, n in enumerate(lengths):
        jitter = (rng.random(ad.shape[1]) - 0.5) * 0.22
        axes[0].scatter(np.full(ad.shape[1], i) + jitter - 0.13, np.abs(ad[i]), s=14,
                        color=style.SERIES[0], alpha=0.75, lw=0,
                        label="autodiff" if i == 0 else None)
        axes[0].scatter(np.full(fd.shape[1], i) + jitter + 0.13, np.abs(fd[i]), s=14,
                        color=style.SERIES[1], alpha=0.75, lw=0,
                        label="finite difference" if i == 0 else None)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(lengths)))
    axes[0].set_xticklabels([str(n) for n in lengths])
    axes[0].set_xlabel("rollout length (days)")
    axes[0].set_ylabel(rf"$|\partial L\,/\,\partial\,${param}$|$ per member")
    axes[0].set_title("(a) every member")
    axes[0].legend(loc="upper left")

    # (b) does the running mean settle on the finite-difference answer?
    running = np.cumsum(ad, axis=1) / members
    reference = fd.mean(axis=1, keepdims=True)
    for i, n in enumerate(lengths):
        axes[1].plot(members, np.abs(running[i] / reference[i]), color=style.SERIES[i],
                     label=f"{n} steps")
    axes[1].axhline(1.0, color=style.INK_MUTED, lw=0.9, ls=(0, (4, 3)))
    axes[1].annotate("ensemble mean = finite difference", (0.97, 0.03), xycoords="axes fraction",
                     ha="right", fontsize=8, color=style.INK_MUTED)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("members averaged")
    axes[1].set_ylabel("running mean AD / mean FD")
    axes[1].set_title("(b) does averaging rescue it?")
    axes[1].legend(loc="upper left", ncols=2)

    axes[2].plot(lengths, np.abs(ad.mean(axis=1)), color=style.SERIES[0], marker="o", label="autodiff")
    axes[2].plot(lengths, np.abs(fd.mean(axis=1)), color=style.SERIES[1], marker="o", label="finite difference")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xticks(lengths)  # the four lengths, not decade ticks that collide
    axes[2].set_xticklabels([str(n) for n in lengths])
    axes[2].minorticks_off()
    axes[2].set_xlabel("rollout length (days)")
    axes[2].set_ylabel(rf"$|$ensemble mean $\partial L\,/\,\partial\,${param}$|$")
    axes[2].set_title("(c) ensemble means")
    axes[2].legend(loc="upper left")

    fig.suptitle(f"{int(d['n_members'])} members, each the settled state plus a "
                 f"{float(d['perturbation']):.0e} K perturbation",
                 fontsize=10.5, y=1.04, color=style.INK)
    fig.tight_layout()
    style.save(fig, common.FIG_DIR / "fig8_ensemble_gradient")


if __name__ == "__main__":
    main()
