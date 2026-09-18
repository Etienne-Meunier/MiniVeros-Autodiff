"""
Figure 1c -- quantitative reproducibility of each gradient method, over rollout length.

Reuses exp1_gradient_check's round-off ensemble (no new simulation): for every
(rollout length, parameter, method) triple we already have N members' gradient
estimates. The spread across members is the ensemble's own standard deviation,
std(members), one number per triple. Three parameter groups (c_k, c_eps, A_h)
are stacked vertically, each a group of three colour stripes -- one per method
(reverse mode, forward mode, finite difference) -- against rollout length.

Columns are one per measured rollout length, evenly spaced and labelled with
the actual day count (not a log-scaled axis) -- there is nothing to interpolate
between them, so the categorical spacing does not imply a false continuum the
way figure 1's log x-axis (a smooth curve between measurements) does.

Each parameter gets its own colour scale (own colourbar at the right, aligned
with that parameter's row group, spanning that parameter's own min/max std
across every method and length): c_k, c_eps and
A_h have unrelated raw gradient magnitudes (different units, different
scales), so a single shared scale would have the biggest-magnitude parameter
dominate the colour range and wash out the other two. With separate per-
parameter scales, a stripe darkening earlier than its neighbours within a
group means that method's estimate stops reproducing across round-off-
perturbed initial conditions sooner than the others, for that parameter --
read timing (left-right position of the transition) within a group, not
colour depth across groups (the three colourbars are not on the same scale).

    python test/paper_figures/fig1c_variance_stripes.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import common
import style

METHODS = [("grad_ad_members", "reverse (AD)"), ("grad_jvp_members", "forward (JVP)"),
           ("grad_fd_members", "finite diff. (FD)")]
GROUP_GAP = 0.55  # rows of empty space between parameter groups
CELL_EDGE = "#fcfcfb"  # style.SURFACE -- thin seam between cells, not a true gridline


def main():
    style.use()
    d = common.load("exp1_gradient_check")
    n, params = d["lengths"], [str(p) for p in d["params"]]
    n_params, n_methods = len(params), len(METHODS)

    CMAP = "turbo"

    std = {key: np.nanstd(d[key], axis=1) for key, _ in METHODS}  # method -> (n_lengths, n_params)
    norms = {}
    for pi, p in enumerate(params):
        stacked = np.concatenate([std[key][:, pi] for key, _ in METHODS])
        finite = stacked[np.isfinite(stacked) & (stacked > 0)]
        norms[p] = LogNorm(vmin=finite.min(), vmax=finite.max())

    x_edges = np.arange(len(n) + 1, dtype=float)  # one evenly spaced column per measured length
    fig, ax = plt.subplots(figsize=(9.2, 0.34 * n_params * n_methods + 1.15))

    group_spans, group_extents, method_ticks, y = [], [], [], 0.0
    param_mesh = {}
    for pi, p in enumerate(params):
        y0 = y
        for key, _ in METHODS:
            row = std[key][:, pi][None, :]  # (1, n_lengths)
            mesh = ax.pcolormesh(x_edges, [y, y + 1.0], row, cmap=CMAP, norm=norms[p],
                                  edgecolors=CELL_EDGE, linewidth=0.5, rasterized=True)
            param_mesh[p] = mesh
            method_ticks.append((y + 0.5, METHODS[len(method_ticks) % n_methods][1]))
            y += 1.0
        group_spans.append(((y0 + y) / 2, p.rsplit(".", 1)[-1]))
        group_extents.append((y0, y, p))
        y += GROUP_GAP
    y -= GROUP_GAP

    ax.set_xlim(x_edges.min(), x_edges.max())
    ax.set_ylim(y, 0)
    ax.set_xticks(x_edges[:-1] + 0.5)
    ax.set_xticklabels([str(int(v)) for v in n], fontsize=7.3, color=style.INK_SOFT, rotation=45, ha="right")
    ax.set_yticks([yy for yy, _ in method_ticks])
    ax.set_yticklabels([lbl for _, lbl in method_ticks], fontsize=7.6, color=style.INK_SOFT)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_xlabel("rollout length (days)")

    for yc, name in group_spans:
        ax.annotate(name, xy=(-0.20, yc), xycoords=("axes fraction", "data"), ha="right", va="center",
                    fontsize=9.5, fontweight="medium", color=style.INK)

    fig.subplots_adjust(right=0.82, top=0.94)
    for y0, y1, p in group_extents:
        frac_top, frac_bottom = 1 - y0 / y, 1 - y1 / y
        cax = ax.inset_axes([1.05, frac_bottom, 0.035, frac_top - frac_bottom])
        cb = fig.colorbar(param_mesh[p], cax=cax, orientation="vertical")
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=2, colors=style.INK_MUTED, labelcolor=style.INK_SOFT, labelsize=7)
        cb.set_label(p.rsplit(".", 1)[-1], color=style.INK_SOFT, fontsize=8.5)

    style.save(fig, common.FIG_DIR / "fig1c_variance_stripes")


if __name__ == "__main__":
    main()
