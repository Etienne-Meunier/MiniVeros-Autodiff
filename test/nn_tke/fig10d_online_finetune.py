"""
Figure 10d -- training the full-function closure: the loss curve.

The centerpiece result: BPTT fine-tuning exp10c's offline-pretrained kappaM_net/diss_net
against exp3's own twin-experiment loss (T,S on the top 3 layers vs the analytic closure's
truth), starting loss (pretrained, not yet fine-tuned online) marked separately from the
training curve.

    python test/nn_tke/fig10d_online_finetune.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common  # nn_tke's own -- must be imported (cached) before the paper_figures path below
               # is added, or it would shadow this one; see fig10b_gradient_check.py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_figures"))

import style


def main():
    style.use()
    d = common.load("exp10d_online_finetune")
    losses, n_steps = d["losses"], int(d["n_steps"])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    it = np.arange(len(losses))
    ax.plot(it, losses, color=style.SERIES[0], lw=1.8)
    ax.scatter([0], [losses[0]], color=style.SERIES[1], s=30, zorder=3, label="offline-pretrained (start)")
    ax.scatter([len(losses) - 1], [losses[-1]], color=style.SERIES[2], s=30, zorder=3, label="after BPTT fine-tune")

    ax.set_yscale("log")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("T,S misfit (top 3 layers) vs analytic-closure truth")
    ax.set_title(f"Training the full-function TKE closure by BPTT ({n_steps}-day rollout)")
    ax.legend(fontsize=8.5)

    improvement = losses[0] / losses[-1]
    ax.annotate(f"{improvement:.1f}x reduction", (len(losses) * 0.6, losses[0] * 0.5),
                fontsize=9, color=style.INK_SOFT)

    fig.tight_layout()
    style.save(fig, common.FIG_DIR / "fig10d_online_finetune")


if __name__ == "__main__":
    main()
