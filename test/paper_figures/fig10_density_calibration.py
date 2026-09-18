"""
Figure 10 -- fitting two hidden mixing parameters against a density-gradient snapshot.

Top row: the loss landscape (log10 loss) with the 3 descent paths drawn on it, one per
starting direction; loss per iteration for the same 3 descents.

Bottom row: the density snapshot for start 3 -- target, then before/after error, as cartopy
Robinson maps styled after the Vercor-Autodiff report's paper_calibration/figures/snapshot.py
(bold title + subtitle, land/coastline, one
shared colour scale across before/after so the shrinkage is the message). Robinson maps are
wide (~2:1), so they get their own full-width row rather than being squeezed into narrow
columns alongside the landscape/convergence panels, which is why this is two rows, not five
panels in one -- matches how Vercor's own snapshot figure is separate from its landscape one.

    python test/paper_figures/fig10_density_calibration.py

Needs cartopy (shapely >= 2.0.5 under numpy 2).
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

import common
import style

SMOOTH_WINDOW = 9  # iterations, odd so the window is centred
PROJECTION = ccrs.Robinson(central_longitude=200)  # Pacific-centred: the seam falls on Africa
DATA_CRS = ccrs.PlateCarree()
LAND_COLOR, NODATA_COLOR = "#cfcfcf", "#f2efe9"


def smoothed_log(y, window=SMOOTH_WINDOW):
    """Centred moving average of log10(y) -- the sign-descent chatter (see the paper's own
    discussion of it) rides on a slowly-drifting trend, and averaging in log space follows that
    trend without letting the chatter's largest values dominate the way a linear-space average
    would."""
    logy = np.log10(y)
    kernel = np.ones(window) / window
    padded = np.pad(logy, window // 2, mode="edge")
    return 10 ** np.convolve(padded, kernel, mode="valid")[:len(y)]


def map_titled(ax, title, subtitle):
    """Bold title + a lighter subtitle above a map axis, as
    paper_calibration/figures/snapshot.py does -- two annotates with generous, explicit
    vertical offsets, safe here because this figure's layout is fully manual (no
    constrained_layout to fight over how much space an annotate needs)."""
    ax.annotate(title, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 24),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=style.INK)
    ax.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8.5, color=style.INK_SOFT)


def geo_colorbar(fig, left, right, fraction, mesh, label, extend):
    """Horizontal colorbar centred under one map (left is right) or spanning two adjacent ones
    -- placed from the axes' drawn positions since the Robinson projection fixes each map's
    aspect, so the axes end up smaller than their gridspec cells (paper_calibration's own
    figures/snapshot.py does the same)."""
    left_box, right_box = left.get_position(), right.get_position()
    width = (right_box.x1 - left_box.x0) * fraction
    cax = fig.add_axes([0.5 * (left_box.x0 + right_box.x1) - width / 2, left_box.y0 - 0.07, width, 0.03])
    fig.colorbar(mesh, cax=cax, orientation="horizontal", extend=extend).set_label(label, fontsize=9)


def main():
    style.use()
    d = common.load("exp10_density_calibration")
    names = [str(p) for p in d["param_names"]]
    true, guesses, fitted = d["true"], d["guesses"], d["fitted"]
    trajectories, losses = d["trajectories"], d["losses"]
    x, y, land = d["x"], d["y"], d["land"]
    xu, yu = d["xu"], d["yu"]
    xu = np.where(xu > 180, xu - 360, xu)  # cartopy's gridliner drops meridians outside [-180, 180]
    short = [n.rsplit(".", 1)[-1] for n in names]  # "tke_closure.c_k" -> "c_k", for compact subtitles

    fig = plt.figure(figsize=(13.5, 9.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.30, left=0.055, right=0.965,
                              top=0.93, bottom=0.05)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.28)
    bottom = outer[1].subgridspec(1, 3, wspace=0.08)

    ax_l, ax_c = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    ax_t, ax_b, ax_a = (fig.add_subplot(bottom[0, i], projection=PROJECTION) for i in range(3))

    floor = np.nanmin(d["landscape"][d["landscape"] > 0]) / 2
    logL = np.log10(np.maximum(d["landscape"], floor))
    mesh = ax_l.contourf(d["c_k"], d["c_eps"], logL.T, levels=24, cmap="viridis")
    ax_l.contour(d["c_k"], d["c_eps"], logL.T, levels=12, colors=style.SURFACE, linewidths=0.5, alpha=0.55)
    style.colorbar(fig, mesh, ax_l, r"$\log_{10}$ loss")

    # The 3 starts sit at one distance from the truth in (log c_k, log c_eps) space --
    # its isoline there is a circle, which maps to this curve in linear (c_k, c_eps).
    dist = float(d["distance"])
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_l.plot(true[0] * np.exp(dist * np.cos(theta)), true[1] * np.exp(dist * np.sin(theta)),
              color=style.INK_MUTED, lw=1.0, ls=(0, (3, 2)), zorder=2, label=f"distance {dist:.2f}")

    for s in range(len(guesses)):
        color = style.SERIES[s]
        traj = trajectories[s]
        ax_l.plot(traj[:, 0], traj[:, 1], color=color, marker="o", ms=3.5, lw=1.6,
                  markevery=max(1, len(traj) // 15), mec=style.SURFACE, mew=0.5,
                  label=f"start {s + 1}", zorder=3)
        ax_l.plot(*guesses[s], marker="o", ms=9, mfc="none", mec=color, mew=1.8, ls="none", zorder=4)
        ax_c.plot(losses[s], color=color, alpha=0.25, lw=1.0, zorder=2)
        ax_c.plot(smoothed_log(losses[s]), color=color, lw=1.8, zorder=3, label=f"start {s + 1}")

    ax_l.plot(*true, marker="*", ms=16, color=style.INK, mec=style.SURFACE, mew=0.6, ls="none",
              label="truth", zorder=5)
    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlim(d["c_k"].min(), d["c_k"].max())
    ax_l.set_ylim(d["c_eps"].min(), d["c_eps"].max())
    style.natural_log_ticks(ax_l, [0.05, 0.075, 0.10, 0.15, 0.20, 0.25], [0.3, 0.4, 0.6, 0.8, 1.0])
    ax_l.set_xlabel(names[0])
    ax_l.set_ylabel(names[1])
    ax_l.set_title("(a) loss landscape")
    ax_l.grid(False)
    ax_l.legend(loc="upper right", labelcolor=style.INK_SOFT, frameon=True,
                facecolor=style.SURFACE, edgecolor="none", framealpha=0.88, fontsize=7.5)

    ax_c.set_yscale("log")
    ax_c.set_xlabel("iteration")
    ax_c.set_ylabel("loss")
    ax_c.set_title("(b) convergence")
    ax_c.legend(loc="upper right")

    # Snapshot: target, then before/after for start 3.
    start = 2
    before = d["density_guess"][start] - d["density_true"]
    after = d["density_fitted"][start] - d["density_true"]
    err_norm = style.symmetric(np.where(land, np.nan, before))
    rms = lambda f: np.sqrt(np.nanmean(np.where(land, np.nan, f) ** 2))
    rms_before, rms_after = rms(before), rms(after)

    mesh_t = ax_t.pcolormesh(x, y, np.where(land, np.nan, d["density_true"]).T, cmap=style.SEQ_WARM,
                             shading="auto", transform=DATA_CRS, rasterized=True)
    map_titled(ax_t, "(c) Target density", f"{short[0]}={true[0]:.3f}, {short[1]}={true[1]:.3f}  (truth)")

    mesh_b = ax_b.pcolormesh(x, y, np.where(land, np.nan, before).T, cmap=style.DIVERGING, norm=err_norm,
                             shading="auto", transform=DATA_CRS, rasterized=True)
    map_titled(ax_b, "(d) Before fit",
               f"{short[0]}={guesses[start][0]:.3f}, {short[1]}={guesses[start][1]:.3f}  ·  rms {rms_before:.1e}")

    mesh_a = ax_a.pcolormesh(x, y, np.where(land, np.nan, after).T, cmap=style.DIVERGING, norm=err_norm,
                             shading="auto", transform=DATA_CRS, rasterized=True)
    map_titled(ax_a, "(e) After fit",
               f"{short[0]}={fitted[start][0]:.3f}, {short[1]}={fitted[start][1]:.3f}  ·  rms {rms_after:.1e}")

    for ax in (ax_t, ax_b, ax_a):
        ax.set_global()
        ax.set_facecolor(NODATA_COLOR)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND_COLOR, edgecolor="none", zorder=2)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555555", linewidth=0.4, zorder=3)
        ax.gridlines(color="#7f7f7f", linewidth=0.25, alpha=0.35, linestyle="-", xlocs=xu, ylocs=yu)
        ax.spines["geo"].set_linewidth(0.6)

    fig.canvas.draw()  # the projection fixes each map's aspect; place colorbars from drawn positions
    geo_colorbar(fig, ax_t, ax_t, 0.7, mesh_t, "kg/m$^3$", "both")
    geo_colorbar(fig, ax_b, ax_a, 0.45, mesh_b, "density error (kg/m$^3$)", "both")

    fitted_strs = ", ".join(f"({f[0]:.3f}, {f[1]:.3f})" for f in fitted)
    fig.suptitle(f"Calibrating ({names[0]}, {names[1]}) against the top {int(d['n_layers'])}-layer "
                 f"density at the end of a {int(d['n_steps'])}-day rollout\n"
                 f"fitted: {fitted_strs}  ·  truth: ({true[0]:.3f}, {true[1]:.3f})",
                 fontsize=10, color=style.INK)
    style.save(fig, common.FIG_DIR / "fig10_density_calibration")


if __name__ == "__main__":
    main()
