"""
Figure 2b -- a local perturbation's footprint, spreading snapshot by snapshot.

One row: surface d(temp at t)/d(point perturbation at t=0), one panel per
measured snapshot, diverging colour centred at zero, sharing one scale
(the last snapshot's, since the footprint only grows). The perturbed point
is marked with a star. Panels are cropped to a regional window around that
point (WINDOW_DEG on each side) -- a single grid point's footprint after a
few weeks is a handful of cells, invisible against the whole globe.

    python test/paper_figures/fig2b_local_perturbation.py
"""

import matplotlib.pyplot as plt
import numpy as np

import common
import style

WINDOW_DEG = 25.0  # crop radius (degrees) around the perturbed point, each panel


def main():
    style.use()
    d = common.load("exp2b_local_perturbation")
    x, y, land = d["x"], d["y"], d["land"]
    steps, sensitivity = d["steps"], d["sensitivity"]  # (n_snapshots, nx, ny, nz)
    i, j = d["point_ij"]
    field = str(d["perturbed_field"])

    n_snap = len(steps)
    surf = sensitivity[..., -1]  # (n_snapshots, nx, ny)
    # shared across every panel and scaled to the largest snapshot (the earliest -- the peak
    # decays monotonically as the perturbation spreads and dilutes), so later panels' wider
    # but fainter footprint is still visible rather than the early ones saturating
    vmax = float(np.nanmax(np.abs(surf)))
    norm = style.symmetric(surf) if vmax > 0 else None

    fig, axes = plt.subplots(1, n_snap, figsize=(2.6 * n_snap, 3.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        mesh = style.map_panel(ax, x, y, surf[k], land, cmap=style.DIVERGING, norm=norm,
                                title=f"+{int(steps[k])} d")
        ax.plot(x[i], y[j], marker="*", ms=9, color=style.INK, mec=style.SURFACE, mew=0.6, zorder=5)
        ax.set_xlim(x[i] - WINDOW_DEG, x[i] + WINDOW_DEG)
        ax.set_ylim(y[j] - WINDOW_DEG, y[j] + WINDOW_DEG)
        if k > 0:
            ax.set_yticklabels([])
    style.colorbar(fig, mesh, axes[-1], r"$\partial\,\mathrm{SST}(t)\ /\ \partial\,\mathrm{point}$")

    fig.suptitle(f"Spread of a unit perturbation to {field} at one grid point", fontsize=10.5, color=style.INK)
    style.save(fig, common.FIG_DIR / "fig2b_local_perturbation")


if __name__ == "__main__":
    main()
