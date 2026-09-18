"""
Figure 0 -- snapshot of the spun-up state: mixed-layer depth and vertically
integrated kinetic energy.

Two maps of the settled global_4deg state every later experiment in this
paper starts from -- what the model actually looks like, before any gradient
or calibration question is asked of it.

Left    mixed-layer depth (de Boyer Montegut density criterion)
Right   vertically integrated kinetic energy, sum_z (u^2 + v^2) dz, u/v
        interpolated to the T grid first

The overlaid mesh is the model's actual T-cell grid (each cell's east/north
face, model.grid.xu/yu) rather than a fixed-interval lat/lon graticule, so the
4-degree resolution (and its latitude-dependent zonal stretching) is visible
directly on the map.

    python test/paper_figures/fig0_spinup_snapshot.py

Needs cartopy (shapely >= 2.0.5 under numpy 2).
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

import common
import style

try:
    import cmocean
    MLD_CMAP, KE_CMAP = cmocean.cm.deep, cmocean.cm.speed
except ImportError:
    MLD_CMAP, KE_CMAP = style.SEQ_BLUE, style.SEQ_WARM

PROJECTION = ccrs.Robinson(central_longitude=200)  # Pacific-centred: the seam falls on Africa
DATA_CRS = ccrs.PlateCarree()
LAND_COLOR, NODATA_COLOR = "#cfcfcf", "#f2efe9"


def titled(axis, title):
    axis.annotate(title, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 10),
                  textcoords="offset points", ha="center", va="bottom",
                  fontsize=12, fontweight="bold", color=style.INK)


def main():
    style.use()
    d = common.load("exp0_spinup_snapshot")
    x, y, land = d["x"], d["y"], d["land"]
    xu, yu = d["xu"], d["yu"]  # real T-cell edges (east/north face), not an arbitrary lat/lon graticule
    xu = np.where(xu > 180, xu - 360, xu)  # cartopy's gridliner drops meridians outside [-180, 180]
    mld = np.where(land, np.nan, d["mld"])
    ke = np.where(land, np.nan, d["ke"])
    mld_high = float(np.nanpercentile(mld, 99))
    ke_high = float(np.nanpercentile(ke, 99))

    fig = plt.figure(figsize=(11.0, 4.6))
    spec = fig.add_gridspec(1, 2, wspace=0.06, left=0.02, right=0.98, top=0.86, bottom=0.16)
    axes = [fig.add_subplot(spec[0, i], projection=PROJECTION) for i in range(2)]

    mld_mesh = axes[0].pcolormesh(x, y, mld.T, cmap=MLD_CMAP, shading="auto", transform=DATA_CRS,
                                   rasterized=True, vmin=0, vmax=mld_high)
    titled(axes[0], "Mixed-layer depth")

    ke_mesh = axes[1].pcolormesh(x, y, ke.T, cmap=KE_CMAP, shading="auto", transform=DATA_CRS,
                                  rasterized=True, vmin=0, vmax=ke_high)
    titled(axes[1], "Vertically integrated kinetic energy")

    for axis in axes:
        axis.set_global()
        axis.set_facecolor(NODATA_COLOR)
        axis.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND_COLOR, edgecolor="none", zorder=2)
        axis.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555555", linewidth=0.4, zorder=3)
        axis.gridlines(color="#7f7f7f", linewidth=0.25, alpha=0.35, linestyle="-", xlocs=xu, ylocs=yu)
        axis.spines["geo"].set_linewidth(0.6)

    # The projection fixes each map's aspect, so the axes end up smaller than their
    # grid cells; place the colour bars from the drawn positions instead.
    fig.canvas.draw()

    def colorbar(axis, mesh, label, extend):
        box = axis.get_position()
        width = (box.x1 - box.x0) * 0.7
        cax = fig.add_axes([0.5 * (box.x0 + box.x1) - width / 2, box.y0 - 0.09, width, 0.035])
        fig.colorbar(mesh, cax=cax, orientation="horizontal", extend=extend).set_label(label)

    colorbar(axes[0], mld_mesh, "MLD (m)", "max")
    colorbar(axes[1], ke_mesh, r"$\sum_z (u^2+v^2)\,\Delta z$  (m$^3$ s$^{-2}$)", "max")

    style.save(fig, common.FIG_DIR / "fig0_spinup_snapshot")


if __name__ == "__main__":
    main()
