"""
Figure 7 -- the checkpointing trade, measured.

(a) wall time for one value-and-gradient against the block size: bigger blocks
    mean more steps recomputed in the backward pass, so time climbs.
(b) the memory the tape adds on top of the process baseline -- and this is the
    panel that corrects the textbook expectation. The classic sqrt(n) optimum
    assumes a stored state and a taped step cost the same. On this model they do
    not: a taped step costs several times a stored state, so the inner term
    dominates, the optimum sits at a block of a few steps rather than at sqrt(n),
    and past that both memory and time simply climb. Large blocks are worse on
    both axes here; `jax.checkpoint` only starts paying when the rollout is long
    enough that storing one state per step is itself the problem.
(c) the gradient each schedule returns. At 80 steps -- inside the window figure 1
    says to trust -- every block size returns the same number, which is what
    "checkpointing is exact" means. At 320 steps they disagree by orders of
    magnitude and by sign. That is not a checkpointing bug: it is figures 1 and 8
    again, seen from a third angle. Past the horizon the gradient is not a
    quantity the code even computes reproducibly.

    python test/paper_figures/fig7_checkpoint_cost.py
"""

import matplotlib.pyplot as plt
import numpy as np

import common
import style


def main():
    style.use()
    d = common.load("exp7_checkpoint_cost")
    rollout, every = d["rollout"], d["checkpoint_every"]
    run, tape, grad = d["run_s"], d["tape_mb"], d["grad"]
    lengths = sorted(set(rollout.tolist()))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    for k, n in enumerate(lengths):
        rows = rollout == n
        colour = style.SERIES[k]
        axes[0].plot(every[rows], run[rows], color=colour, marker="o", label=f"{n} steps")
        axes[1].plot(every[rows], tape[rows], color=colour, marker="o", label=f"{n} steps")

        # Panel (c): each schedule's gradient relative to the one at the finest blocking,
        # which is the least-recomputed and so the natural reference.
        reference = grad[rows][0]
        axes[2].plot(every[rows], np.abs(grad[rows] / reference), color=colour, marker="o",
                     label=f"{n} steps")

    axes[0].set_ylabel("value + gradient (s)")
    axes[0].set_title("(a) time")
    axes[1].set_ylabel("peak memory above baseline (MB)")
    axes[1].set_title("(b) memory the tape adds")
    axes[2].set_yscale("log")
    axes[2].axhline(1.0, color=style.INK_MUTED, lw=0.9, ls=(0, (4, 3)))
    axes[2].set_ylabel("gradient / gradient at the finest blocking")
    axes[2].set_title("(c) is it the same answer?")
    axes[2].annotate("exact", (0.03, 0.06), xycoords="axes fraction", fontsize=8, color=style.INK_MUTED)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("checkpoint_every (steps per block)")
        ax.legend(loc="upper left")
    # Mark where memory actually bottoms out, and sqrt(n) for contrast -- they differ.
    for k, n in enumerate(lengths):
        rows = rollout == n
        best = every[rows][int(np.argmin(tape[rows]))]
        axes[1].axvline(best, color=style.SERIES[k], lw=1.0, ls="-", alpha=0.45)
        axes[1].axvline(np.sqrt(n), color=style.SERIES[k], lw=0.9, ls=(0, (4, 3)), alpha=0.45)
    axes[1].annotate("solid: measured minimum\ndashed: $\\sqrt{n}$", (0.97, 0.06),
                     xycoords="axes fraction", ha="right", fontsize=8, color=style.INK_MUTED)

    fig.suptitle("Cost of one gradient, and whether the schedule changes the answer",
                 fontsize=10.5, y=1.04, color=style.INK)
    fig.tight_layout()
    style.save(fig, common.FIG_DIR / "fig7_checkpoint_cost")


if __name__ == "__main__":
    main()
