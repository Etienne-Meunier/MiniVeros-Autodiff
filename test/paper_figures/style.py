"""
One look for all five figures: palette, colormaps, and the two drawing helpers
(a map panel and a figure writer) the fig*.py scripts share.

Colours come from a validated palette: a fixed categorical order for series, a
single-hue ramp for magnitudes, and a two-hue blue/red ramp with a neutral grey
midpoint for signed quantities (differences, sensitivities). Sequential ramps are
never rainbow, and the diverging midpoint is never a hue.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.ticker import NullFormatter, ScalarFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LAND = "#dedcd3"

# Categorical slots, assigned in this order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Magnitude: one hue, light -> dark. Warm ramp for temperature, blue for everything else.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
)
SEQ_WARM = LinearSegmentedColormap.from_list(
    "seq_warm", ["#fdece2", "#f9c7a9", "#f2996b", "#eb6834", "#b8451a", "#71260a"]
)
# Polarity: blue <-> red across a neutral grey midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0aeae", "#e34948", "#8f1f1f"]
)
_LAND_CMAP = ListedColormap([LAND])


def use():
    """Apply the shared rcParams. Call once at the top of a fig*.py script."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.titlesize": 9.5,
        "axes.titleweight": "medium",
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SOFT,
        "ytick.labelcolor": INK_SOFT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


def symmetric(field):
    """A TwoSlopeNorm centred on zero, scaled to the field -- for diverging maps."""
    v = float(np.nanmax(np.abs(field)))
    v = v if v > 0 else 1.0
    return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)


def map_panel(ax, x, y, field, land, cmap=SEQ_WARM, norm=None, title=None, **kwargs):
    """Draw one horizontal field with land greyed out. `field` is (nx, ny)."""
    ax.pcolormesh(x, y, np.where(land, 1.0, np.nan).T, cmap=_LAND_CMAP, vmin=0, vmax=1, rasterized=True)
    mesh = ax.pcolormesh(
        x, y, np.where(land, np.nan, field).T, cmap=cmap, norm=norm, shading="auto", rasterized=True, **kwargs
    )
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect("equal")
    ax.grid(False)
    ax.tick_params(length=2)
    if title:
        ax.set_title(title, pad=4)
    return mesh


def colorbar(fig, mesh, ax, label, **kwargs):
    """A thin colourbar with a muted label, sized to the panel it belongs to."""
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, **kwargs)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, colors=INK_MUTED, labelcolor=INK_SOFT, labelsize=8)
    cb.set_label(label, color=INK_SOFT, fontsize=8.5)
    return cb


def natural_log_ticks(ax, xticks, yticks):
    """On log-scaled x/y axes, show the given tick values in plain decimal (e.g. "0.10", not
    "10^-1") -- log positioning is what makes a fixed distance in log-parameter space read as
    a fixed screen distance, but physicists reading the axis want the parameter's own units."""
    plain = ScalarFormatter()
    plain.set_scientific(False)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(NullFormatter())
    plain_y = ScalarFormatter()
    plain_y.set_scientific(False)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(plain_y)
    ax.yaxis.set_minor_formatter(NullFormatter())


def label_series_ends(ax, x, series, min_gap=0.07):
    """Direct-label each series just past its last finite point, nudged apart when labels collide.

    `series` is a list of (y, text, color). Call after the axes scales and limits are set,
    since the nudging is done in axes coordinates.
    """
    to_axes = ax.transAxes.inverted().transform
    placed = []
    for y, text, color in series:
        finite = np.flatnonzero(np.isfinite(y))
        if len(finite):
            i = finite[-1]
            placed.append([to_axes(ax.transData.transform((x[i], y[i])))[1], text, color])

    placed.sort(key=lambda item: item[0])
    for k in range(1, len(placed)):
        placed[k][0] = max(placed[k][0], placed[k - 1][0] + min_gap)
    overflow = placed[-1][0] - 1.0 if placed else 0.0
    if overflow > 0:  # the stack grew past the top of the panel; slide it back down
        for item in placed:
            item[0] -= overflow

    for height, text, color in placed:
        ax.annotate(text, (1.02, height), xycoords="axes fraction", va="center",
                    fontsize=8.5, color=color, fontweight="medium", annotation_clip=False)


def save(fig, path):
    """Write a figure as png, creating the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {path.with_suffix('.png')}")
