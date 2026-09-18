"""
Figure 10b -- does the H=80-day horizon survive going from 2 scalars to ~700 NN weights?

One panel: reverse-mode-vs-finite-difference relative error (random-direction projection, see
exp10b_gradient_check.py) against rollout length -- smoothed median across N_DIRECTIONS random
directions in the full (both nets') weight space, with each direction plotted as a small cross.
The scalar closure's own horizon (H=80d, exp1_gradient_check.py) is drawn alongside for direct
comparison.

    python test/nn_tke/fig10b_gradient_check.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common  # nn_tke's own -- must be imported (and thus cached in sys.modules) before the
               # paper_figures path below is added, or it would shadow this one

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_figures"))

import style
from fig1_gradient_check import add_steps_axis, smooth_curve


def main():
    style.use()
    d = common.load("exp10b_gradient_check")
    n = d["lengths"]
    ad_proj, fd_proj = d["ad_proj"], d["fd_proj"]  # (length, direction)
    horizon_nn, horizon_scalar = int(d["horizon_days"]), int(d["scalar_horizon_days"])
    dt = float(d["dt_tracer"])

    rel_err = np.abs(ad_proj - fd_proj) / np.abs(fd_proj)  # (length, direction)
    med = np.nanmedian(rel_err, axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x_fine, y_fine = smooth_curve(n, med)
    ax.plot(x_fine, y_fine, color=style.SERIES[0], lw=1.8, alpha=0.75, zorder=2, label="NN closure (~700 weights)")
    ax.scatter(n, med, color=style.SERIES[0], s=22, zorder=3)
    n_repeated = np.repeat(n, rel_err.shape[1])
    ax.scatter(n_repeated, rel_err.ravel(), color=style.SERIES[0], s=18, marker="x", lw=0.8, zorder=4)

    ax.axvline(horizon_nn, color=style.SERIES[0], lw=0.9, ls=(0, (1, 2)))
    ax.axvline(horizon_scalar, color=style.INK_MUTED, lw=0.9, ls=(0, (4, 2)))
    ax.annotate(f"H_nn = {horizon_nn} d", (horizon_nn, med.max()), textcoords="offset points",
                xytext=(4, -4), fontsize=8, color=style.SERIES[0], va="top")
    ax.annotate(f"H (scalar) = {horizon_scalar} d", (horizon_scalar, med.max()), textcoords="offset points",
                xytext=(4, -16), fontsize=8, color=style.INK_MUTED, va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(n.min() * 0.85, n.max() * 1.2)
    ax.set_xlabel("rollout length (days)")
    ax.set_ylabel(r"$|g_{\rm AD}\!\cdot\!d - g_{\rm FD}\!\cdot\!d|\ /\ |g_{\rm FD}\!\cdot\!d|$")
    ax.set_title("Coefficient-mode NN closure (~700 weights): reverse mode vs finite difference, random directions")
    add_steps_axis(ax, dt)

    fig.tight_layout()
    style.save(fig, common.FIG_DIR / "fig10b_gradient_check")


if __name__ == "__main__":
    main()
