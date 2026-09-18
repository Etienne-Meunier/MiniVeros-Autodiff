"""
Figure 10e -- does the full-function closure's BPTT fit still "succeed" past the NN horizon?

exp4_calibration_limit.py's own question, now for the NN closure: final loss (start vs end of
each length's fine-tune) against rollout length, with H_nn=80d (exp10b_gradient_check.py)
marked. The relative improvement shrinking with length, without ever diverging even past H_nn,
is itself the finding worth showing -- optimization keeps "working" in the sense of reducing
the loss well past the point where its gradients are known to be trustworthy.

    python test/nn_tke/fig10e_calibration_limit.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common  # nn_tke's own -- see fig10b_gradient_check.py for why this import order matters

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_figures"))

import style


def main():
    style.use()
    d = common.load("exp10e_calibration_limit")
    lengths = d["lengths"]
    loss_start, loss_end = d["loss_start"], d["loss_end"]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(lengths, loss_start, color=style.SERIES[1], marker="o", ms=5, label="start (pretrained, untrained-online)")
    ax.plot(lengths, loss_end, color=style.SERIES[0], marker="o", ms=5, label="end (after 200-iter BPTT fine-tune)")

    horizon = 80
    if lengths.min() <= horizon <= lengths.max():
        ax.axvline(horizon, color=style.INK_MUTED, lw=0.9, ls=(0, (1, 2)))
        ax.annotate("H_nn = 80 d", (horizon, loss_start.max()), textcoords="offset points",
                    xytext=(4, -4), fontsize=8, color=style.INK_MUTED, va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rollout length (days)")
    ax.set_ylabel("T,S misfit (top 3 layers) vs analytic-closure truth")
    ax.set_title("Full-function closure: BPTT fine-tune loss, before/after, by length")
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    style.save(fig, common.FIG_DIR / "fig10e_calibration_limit")


if __name__ == "__main__":
    main()
