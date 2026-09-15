"""
Reads the 5 full-state runs written by run_top5_full_state.py and writes two
GIFs to report/ck_ceps_density_figures/: top5_mld_evolution.gif (mixed-layer
depth) and top5_temp_evolution.gif (surface temperature), each with 5 panels
(one per run) animated across the 30-year run.

Frames are subsampled (every FRAME_STRIDE-th of the 365 logged snapshots) to
keep the gif a reasonable size; vmin/vmax are shared across the 5 panels but
recomputed per frame (matching test/plot_matrix_report.py's convention) so
the cold start isn't washed out by the final range. Colorbar sits in a fixed
cax computed once outside the frame loop -- letting fig.colorbar auto-shrink
the panels frame by frame (its default behaviour with ax=<list>) reflows
their width slightly differently each time and both flickers and can
overlap the last panel; a fixed cax never moves.

    python test/ck_ceps_sweep/fig_state_gif.py

Never re-runs a simulation.
"""

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import common

sys.path.append(str(Path(__file__).resolve().parents[1] / "paper_figures"))
import style  # noqa: E402
from mini_veros.core.density.get_rho import get_potential_rho
from mini_veros.setups.acc import full

from fig_mld_snapshots import mld_from_density

style.use()

FRAME_STRIDE = 5  # every 5th logged snapshot (~150 days apart) -> 73 frames


def load_frames(field, zt, eq_of_state_type):
    """Per (c_k, c_eps): (n_frames, nx, ny) array for `field` ('temp' surface, or 'mld')."""
    out = []
    for c_k, c_eps in common.TOP5:
        path = common.DATA_DIR / "full_state" / f"ck{c_k:.4g}_eps{c_eps:.4g}.npz"
        d = np.load(path)
        temp, salt = d["temp"][::FRAME_STRIDE, 2:-2, 2:-2, :], d["salt"][::FRAME_STRIDE, 2:-2, 2:-2, :]
        if field == "temp":
            out.append(temp[..., -1])
        elif field == "mld":
            rho = get_potential_rho(eq_of_state_type, salt, temp, 0.0)
            out.append(mld_from_density(rho, zt))
    return out


def make_gif(field, x, y, land, out_path, cmap, n_frames, years):
    zt = None
    ref_zt = np.asarray(full.build({})[0].grid.zt) if field == "mld" else None
    frames_by_run = load_frames(field, ref_zt, 3)
    titles = [f"c_k={c_k:.3g}, c_eps={c_eps:.3g}" for c_k, c_eps in common.TOP5]

    fig, axes = style.plt.subplots(1, 5, figsize=(16.5, 3.6))
    fig.subplots_adjust(left=0.03, right=0.93, top=0.82, bottom=0.1, wspace=0.15)
    cax = fig.add_axes([0.955, 0.12, 0.012, 0.62])

    gif_frames = []
    for i in range(n_frames):
        frame_stack = [f[i] for f in frames_by_run]
        vmin = min(np.nanmin(np.where(land, np.nan, f)) for f in frame_stack)
        vmax = max(np.nanmax(np.where(land, np.nan, f)) for f in frame_stack)
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0

        for ax in axes:
            ax.clear()
        cax.clear()
        for ax, f, title in zip(axes, frame_stack, titles):
            mesh = style.map_panel(ax, x, y, f, land, cmap=cmap, vmin=vmin, vmax=vmax, title=title)
        fig.colorbar(mesh, cax=cax)
        fig.suptitle(f"{field}  (model year {years[i]:.1f})")
        fig.canvas.draw()
        gif_frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())

    style.plt.close(fig)
    imageio.mimsave(out_path, gif_frames, duration=0.15, loop=0)
    print(f"wrote {out_path} ({n_frames} frames)")


def main():
    ref_model, _, _ = full.build({})
    x = np.asarray(ref_model.grid.xt[2:-2])
    y = np.asarray(ref_model.grid.yt[2:-2])
    land = np.asarray(ref_model.boundary_conditions.maskT[2:-2, 2:-2, -1]) == 0

    n_logs = np.load(common.DATA_DIR / "full_state" / f"ck{common.TOP5[0][0]:.4g}_eps{common.TOP5[0][1]:.4g}.npz")["temp"].shape[0]
    n_frames = len(range(0, n_logs, FRAME_STRIDE))
    years = (np.arange(0, n_frames * FRAME_STRIDE, FRAME_STRIDE) + 1) * common.LOG_EVERY_DAYS / 365.0

    common.FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_gif("mld", x, y, land, common.FIG_DIR / "top5_mld_evolution.gif", "turbo", n_frames, years)
    make_gif("temp", x, y, land, common.FIG_DIR / "top5_temp_evolution.gif", "turbo", n_frames, years)


if __name__ == "__main__":
    main()
