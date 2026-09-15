"""
Figure 6 -- where the gradient spikes come from.

Four panels: (a) how often autodiff disagrees sharply with finite
differences, as a histogram of the ratio between them across an ensemble of
start dates, at two rollout lengths; (b) where that disagreement is building,
read off a manual step-by-step reverse sweep -- the norm of the adjoint of
each prognostic field, backward step by backward step, for one spiking and one
normal start date; (c) the same ensemble's spike fraction after switching off
one likely source of non-smoothness at a time; (d) the map location of
|adjoint of temp|'s maximum, backward step by backward step, for the spiking
member -- where panel (b)'s growth is concentrated.

    python test/paper_figures/fig6_gradient_spikes.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import common
import style


def main():
    style.use()
    d = common.load("exp6_gradient_spikes")
    lengths = d["lengths"]
    field_names = [str(f) for f in d["field_names"]]
    x, y, land = d["x"], d["y"], d["land"]

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.0))

    # (a) histogram of AD/FD ratio, one length per colour
    ax = axes[0]
    for k, n in enumerate(lengths):
        r = d[f"ratios__{n}"]
        ax.hist(np.log10(np.abs(r)), bins=20, histtype="step", color=style.SERIES[k], lw=1.6,
               label=f"{int(n)} d")
    ax.axvline(0.0, color=style.INK_MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.set_xlabel(r"$\log_{10}|g_{\rm AD}/g_{\rm FD}|$")
    ax.set_ylabel("members")
    ax.set_title("(a) how often")
    ax.legend()

    # (b) adjoint norm per field, backward step by backward step
    ax = axes[1]
    norms_spike, norms_normal = d["norms_spike"], d["norms_normal"]  # (n_steps, n_fields)
    n_steps = norms_spike.shape[0]
    backward_step = np.arange(n_steps, 0, -1)
    for j, f in enumerate(field_names):
        ax.plot(backward_step, norms_spike[:, j], color=style.SERIES[j], lw=1.6, label=f)
        ax.plot(backward_step, norms_normal[:, j], color=style.SERIES[j], lw=1.0, ls=(0, (2, 1.5)), alpha=0.6)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("backward step (steps before the end)")
    ax.set_ylabel(r"$\|\partial L/\partial(\text{field})\|$")
    ax.set_title("(b) where -- adjoint norm per field\n(solid: spiking member, dashed: normal member)")
    ax.legend(fontsize=7.5, ncols=2)

    # (c) ablation spike fractions
    ax = axes[2]
    names, frac = [str(n) for n in d["ablation_names"]], d["ablation_spike_fraction"]
    ax.bar(range(len(names)), frac, color=style.SERIES[: len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("spike fraction")
    ax.set_title("(c) why -- ablations")

    # (d) map trail of the spiking member's peak |adjoint of temp|, backward step by backward step
    ax = axes[3]
    peak_spike = d["peak_spike"]  # (n_steps, 3): (ix, iy, iz)
    style.map_panel(ax, x, y, np.full((len(x), len(y)), np.nan), land, cmap=style.SEQ_BLUE)
    xs, ys = x[peak_spike[:, 0]], y[peak_spike[:, 1]]
    sc = ax.scatter(xs, ys, c=backward_step, cmap=style.SEQ_BLUE, s=18,
                     norm=LogNorm(vmin=1, vmax=backward_step.max()), edgecolors="none")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("(d) where -- peak |adjoint of temp|\n(spiking member)")
    style.colorbar(fig, sc, ax, "backward step")

    fig.suptitle(f"Gradient spikes: frequency, location, and cause "
                 f"(param {str(d['param'])}, {int(d['n_members'])} members, {int(d['spacing_days'])} d apart)",
                 fontsize=10.5, y=1.03, color=style.INK)
    fig.tight_layout()
    style.save(fig, common.FIG_DIR / "fig6_gradient_spikes")


if __name__ == "__main__":
    main()
