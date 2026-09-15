"""
Reads the 5 full-state runs written by run_top5_full_state.py
(DATA_DIR/full_state/ck{c_k}_eps{c_eps}.npz) and writes
report/ck_ceps_density_figures/fig_mld_snapshots.{pdf,png}: one map per run of
the mixed-layer depth (MLD) anomaly -- MLD minus the 5-run mean MLD at each
cell -- averaged over the final model-year (last 12 of the 365 logged monthly
snapshots). Anomaly rather than raw MLD makes the (c_k, c_eps) effect the
whole colormap range, instead of it being a small wiggle on top of the
channel's much larger shared MLD pattern.

MLD per snapshot/column: depth where the potential density first exceeds the
surface value by 0.03 kg/m^3 (the de Boyer Montegut density criterion),
scanning down from the surface; a column that never crosses the threshold is
reported as fully mixed to the bottom. Averaging MLD *after* computing it
per-snapshot (rather than computing one MLD from a time-averaged density
profile) avoids smoothing out the threshold crossing.

Never re-runs a simulation.
"""

import sys
from pathlib import Path

import numpy as np

import common

sys.path.append(str(Path(__file__).resolve().parents[1] / "paper_figures"))
import style  # noqa: E402
from mini_veros.core.density.get_rho import get_potential_rho
from mini_veros.setups.acc import full

try:
    import cmocean
    MEAN_CMAP = cmocean.cm.deep
except ImportError:
    MEAN_CMAP = style.SEQ_BLUE

style.use()

MLD_THRESHOLD = 0.03  # kg/m^3, de Boyer Montegut et al. 2004 density criterion


def mld_from_density(rho, zt):
    """rho: (..., nz) potential density, index 0=bottom .. nz-1=surface (see model conventions).
    Returns (...,) MLD depth in positive meters, averaged over nothing -- one map per leading dims.
    """
    rev = rho[..., ::-1]              # index 0=surface .. nz-1=bottom
    zt_rev = zt[::-1]
    exceed = (rev - rev[..., :1]) >= MLD_THRESHOLD
    any_exceed = exceed.any(axis=-1)
    first_idx = np.where(any_exceed, np.argmax(exceed, axis=-1), len(zt) - 1)
    return -zt_rev[first_idx]


def main():
    ref_model, _, _ = full.build({})
    x = np.asarray(ref_model.grid.xt[2:-2])
    y = np.asarray(ref_model.grid.yt[2:-2])
    land = np.asarray(ref_model.boundary_conditions.maskT[2:-2, 2:-2, -1]) == 0
    eq_of_state_type = ref_model.config.eq_of_state_type

    mld_maps, titles = [], []
    for c_k, c_eps in common.TOP5:
        path = common.DATA_DIR / "full_state" / f"ck{c_k:.4g}_eps{c_eps:.4g}.npz"
        d = np.load(path)
        temp, salt, zt = d["temp"][-common.KEEP_LAST:], d["salt"][-common.KEEP_LAST:], d["zt"]
        rho = get_potential_rho(eq_of_state_type, salt, temp, 0.0)  # (KEEP_LAST, nx+4, ny+4, nz)
        mld = mld_from_density(rho[:, 2:-2, 2:-2, :], zt)           # (KEEP_LAST, nx, ny)
        mld_maps.append(mld.mean(axis=0))
        titles.append(f"c_k={c_k:.3g}, c_eps={c_eps:.3g}")

    reference = np.mean(np.stack(mld_maps), axis=0)  # 5-run mean MLD, per cell
    anomaly_maps = [m - reference for m in mld_maps]

    stacked = np.stack([np.where(land, np.nan, a) for a in anomaly_maps])
    norm = style.symmetric(stacked)  # TwoSlopeNorm centred on 0, scaled to the largest anomaly

    # A dedicated spacer column (not an inset) holds the mean panel's own colorbar --
    # an inset stuck out past axes[0]'s right edge into axes[1]'s own space and its
    # tick labels were painted over when axes[1]'s data was drawn afterwards.
    n = len(anomaly_maps)
    fig = style.plt.figure(figsize=(3.1 * (n + 1) + 0.9, 4.2))
    gs = style.matplotlib.gridspec.GridSpec(1, n + 2, width_ratios=[1, 0.1] + [1] * n, wspace=0.45, figure=fig)
    ax_mean = fig.add_subplot(gs[0, 0])
    cax_mean = fig.add_subplot(gs[0, 1])
    axes = [ax_mean] + [fig.add_subplot(gs[0, i + 2], sharey=ax_mean) for i in range(n)]

    mean_mesh = style.map_panel(ax_mean, x, y, reference, land, cmap=MEAN_CMAP, title="5-run mean")
    mean_cb = fig.colorbar(mean_mesh, cax=cax_mean, label="MLD (m)")
    mean_cb.ax.yaxis.set_major_locator(style.matplotlib.ticker.MaxNLocator(nbins=5))
    mean_cb.ax.tick_params(labelsize=8)

    for ax, anomaly, title in zip(axes[1:], anomaly_maps, titles):
        mesh = style.map_panel(ax, x, y, anomaly, land, cmap=style.DIVERGING, norm=norm, title=title)
    axes[0].set_ylabel("latitude")
    for ax in axes:
        ax.set_xlabel("longitude")
    fig.suptitle(f"mixed-layer depth: 5-run mean, and each run's anomaly from it, final-model-year "
                 f"average (threshold {MLD_THRESHOLD} kg/m^3)", y=1.03)
    style.colorbar(fig, mesh, axes[1:], "MLD anomaly (m)")

    out_path = common.FIG_DIR / "fig_mld_snapshots"
    common.FIG_DIR.mkdir(parents=True, exist_ok=True)
    style.save(fig, out_path)


if __name__ == "__main__":
    main()
