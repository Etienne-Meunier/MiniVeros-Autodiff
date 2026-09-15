"""
Figure 1b -- does the horizon hold for other fields, and do jvp/vjp agree with each other?

Two panels:
  (a) reverse-mode-vs-finite-difference relative error against rollout
      length, one line per (field, parameter) pair -- colour is the
      parameter (same palette as figure 1), solid is temperature, dashed is
      salinity. If every line crosses figure 1's horizon at about the same
      length, the horizon is a property of the rollout, not of which field
      the loss happened to look at.
  (b) forward mode (jvp) vs reverse mode (vjp . direction) relative
      disagreement against length, no finite difference involved: smoothed
      median across N_DIRECTIONS random directions in parameter space, with
      each direction plotted as a small cross. This is the two
      exact-arithmetic-equivalent autodiff modes checked against each other.

    python test/paper_figures/fig1b_field_jvp_vjp.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import common
import style
from fig1_gradient_check import add_steps_axis, smooth_curve

FIELD_STYLE = {"temp": "-", "salt": (0, (5, 1.5))}


def main():
    style.use()
    d = common.load("exp1b_field_jvp_vjp")
    n = d["lengths"]
    fields, params, dt = [str(f) for f in d["fields"]], [str(p) for p in d["params"]], float(d["dt_tracer"])
    grad_ad, grad_fd = d["grad_ad"], d["grad_fd"]  # (length, field, param)
    jvp_vals, vjp_proj = d["jvp_vals"], d["vjp_proj"]  # (length, direction)

    rel_err = np.abs(grad_ad - grad_fd) / np.abs(grad_fd)  # (length, field, param)
    rel_err_modes = np.abs(jvp_vals - vjp_proj) / np.abs(vjp_proj)  # (length, direction)

    fig = plt.figure(figsize=(11.5, 3.6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    for fi, field_name in enumerate(fields):
        for pi, p in enumerate(params):
            x_fine, y_fine = smooth_curve(n, rel_err[:, fi, pi])
            ax_a.plot(x_fine, y_fine, color=style.SERIES[pi], ls=FIELD_STYLE.get(field_name, "-"),
                      label=f"{p} ({field_name})" if fi == 0 else None)
            ax_a.scatter(n, rel_err[:, fi, pi], color=style.SERIES[pi], s=14, zorder=3)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(n.min() * 0.85, n.max() * 1.2)
    ax_a.set_xlabel("rollout length (days)")
    ax_a.set_ylabel(r"$|g_{\rm AD} - g_{\rm FD}|\ /\ |g_{\rm FD}|$")
    ax_a.set_title("(a) reverse mode vs finite difference, by field and parameter")
    handles = [plt.Line2D([], [], color=style.SERIES[pi], label=p) for pi, p in enumerate(params)]
    handles += [plt.Line2D([], [], color=style.INK_MUTED, ls=ls, label=fname)
                for fname, ls in FIELD_STYLE.items()]
    ax_a.legend(handles=handles, fontsize=7.5, loc="lower right", ncols=2)

    med = np.nanmedian(rel_err_modes, axis=1)
    x_fine, y_fine = smooth_curve(n, med)
    ax_b.plot(x_fine, y_fine, color=style.SERIES[0], lw=1.8, alpha=0.7, zorder=2)
    ax_b.scatter(n, med, color=style.SERIES[0], s=20, alpha=0.7, zorder=3)
    n_repeated = np.repeat(n, rel_err_modes.shape[1])
    ax_b.scatter(n_repeated, rel_err_modes.ravel(), color=style.SERIES[0], s=18, marker="x", lw=0.8, zorder=4)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(n.min() * 0.85, n.max() * 1.2)
    ax_b.set_xlabel("rollout length (days)")
    ax_b.set_ylabel(r"$|jvp - vjp\!\cdot\!direction|\ /\ |vjp\!\cdot\!direction|$")
    ax_b.set_title("(b) forward mode vs reverse mode (no finite difference)")

    for ax in (ax_a, ax_b):
        add_steps_axis(ax, dt)

    fig.suptitle("Generalising figure 1: other fields, and forward vs reverse mode directly",
                 fontsize=10.5, y=1.08, color=style.INK)
    fig.tight_layout()
    style.save(fig, common.FIG_DIR / "fig1b_field_jvp_vjp")


if __name__ == "__main__":
    main()
