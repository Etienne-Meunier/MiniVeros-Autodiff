"""
Figure 1 -- autodiff vs finite differences over increasing rollout length.

Two panels, both labelled in days (bottom) and steps (top), both a smooth
curve (a cubic B-spline through log(length)) with a dot at every length that
was actually measured -- nothing between the dots is real data:
  (a) relative disagreement between reverse-mode and finite differences, per
      parameter, solid -- this defines the horizon. The same disagreement for
      forward-mode (jvp) vs finite differences is overlaid dotted, same
      colour per parameter: reverse and forward mode differentiate the exact
      same computation, so where the dotted line tracks the solid one, the
      instability is a property of the rollout, not of which autodiff mode
      walked it.
  (b) for c_k, the round-off ensemble's gradient against length: the smoothed
      median as a line with a dot at each measured length, every individual
      member plotted as a small cross around it, now for all three of
      reverse mode, forward mode and the finite difference. The scatter is
      the round-off ensemble's own reproducibility; where the lines sit (and
      diverge from each other) is the same signed-value story the old
      separate panel told.

    python test/paper_figures/fig1_gradient_check.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline

import common
import style

YEAR = 365 * 86400
SWARM_PARAM = "c_k"
LINTHRESH = 0.1  # panel (b)'s symlog linear region, and the signed-log space smoothing happens in


def add_steps_axis(ax, dt_tracer):
    """A second x axis on top of `ax`, in steps rather than days."""
    steps_per_day = 86400.0 / dt_tracer
    ax_steps = ax.secondary_xaxis("top", functions=(lambda d: d * steps_per_day, lambda s: s / steps_per_day))
    ax_steps.set_xlabel("steps", fontsize=8.5, color=style.INK_MUTED)
    ax_steps.tick_params(labelsize=8, colors=style.INK_MUTED)


def slog(v, linthresh=LINTHRESH):
    """sign(v) * log10(1 + |v| / linthresh) -- matches the symlog display, so smoothing in
    this space looks smooth once plotted, instead of being dominated by whichever endpoint
    of a segment happens to be orders of magnitude larger.
    """
    return np.sign(v) * np.log10(1 + np.abs(v) / linthresh)


def inv_slog(t, linthresh=LINTHRESH):
    return np.sign(t) * linthresh * (10.0 ** np.abs(t) - 1)


def smooth_curve(x, y, transform=np.log10, inverse=lambda t: 10.0**t, n_fine=300):
    """A cubic B-spline through (x, transform(y)), evaluated on a fine log-spaced grid and
    mapped back with `inverse`.
    """
    lx = np.log10(x)
    k = min(3, len(x) - 1)
    spline = make_interp_spline(lx, transform(y), k=k)
    lx_fine = np.linspace(lx.min(), lx.max(), n_fine)
    return 10.0**lx_fine, inverse(spline(lx_fine))


def main():
    style.use()
    d = common.load("exp1_gradient_check")
    n, params, dt = d["lengths"], [str(p) for p in d["params"]], float(d["dt_tracer"])
    ad, fd = d["grad_ad"], d["grad_fd"]  # (length, param) -- member 0, the unperturbed state
    jvp = d["grad_jvp"] if "grad_jvp" in d else None
    ad_members, fd_members = d["grad_ad_members"], d["grad_fd_members"]  # (length, member, param)
    jvp_members = d["grad_jvp_members"] if "grad_jvp_members" in d else None
    horizon_days = int(d["horizon_days"])
    j = params.index(SWARM_PARAM)

    rel_err = np.abs(ad - fd) / np.abs(fd)  # (length, param)
    rel_err_jvp = np.abs(jvp - fd) / np.abs(fd) if jvp is not None else None  # (length, param)

    fig = plt.figure(figsize=(13.0, 3.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 2], wspace=0.28)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # (a) relative disagreement, one line per parameter -- smoothed, with a dot per measurement.
    # forward-mode (jvp) vs finite difference is overlaid dotted, same colour, no extra scatter --
    # a second reference line, not a second dataset to read dot-by-dot.
    for k, p in enumerate(params):
        x_fine, y_fine = smooth_curve(n, rel_err[:, k])
        ax_a.plot(x_fine, y_fine, color=style.SERIES[k], label=p)
        ax_a.scatter(n, rel_err[:, k], color=style.SERIES[k], s=16, zorder=3)
        if rel_err_jvp is not None:
            x_fine_j, y_fine_j = smooth_curve(n, rel_err_jvp[:, k])
            ax_a.plot(x_fine_j, y_fine_j, color=style.SERIES[k], ls=(0, (1, 1)), lw=1.3, alpha=0.85)
    ax_a.axvline(horizon_days, color=style.INK_MUTED, lw=0.9, ls=(0, (1, 2)))
    ax_a.annotate(f"H = {horizon_days} d", (horizon_days, rel_err.max()), textcoords="offset points",
                  xytext=(4, -4), fontsize=8, color=style.INK_MUTED, va="top")
    ax_a.set_title("(a) autodiff vs finite difference")
    ax_a.set_ylabel(r"$|g_{\rm AD} - g_{\rm FD}|\ /\ |g_{\rm FD}|$")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(n.min() * 0.85, n.max() * 1.2)
    ax_a.set_xlabel("rollout length (days)")
    style.label_series_ends(ax_a, n, [(rel_err[:, k], p, style.SERIES[k]) for k, p in enumerate(params)])
    if rel_err_jvp is not None:
        ax_a.plot([], [], color=style.INK_MUTED, ls="-", lw=1.5, label="reverse mode (ad)")
        ax_a.plot([], [], color=style.INK_MUTED, ls=(0, (1, 1)), lw=1.3, label="forward mode (jvp)")
        ax_a.legend(loc="lower right", fontsize=7.5, handlelength=1.6)

    # (b) round-off ensemble for c_k: smoothed median line + a dot per measured length,
    # every individual member as a small cross in place of the old shaded band
    series_b = [(ad_members, style.SERIES[0], "reverse mode (ad)"), (fd_members, style.SERIES[1], "finite difference")]
    if jvp_members is not None:
        series_b.append((jvp_members, style.SERIES[3], "forward mode (jvp)"))
    for members, color, label in series_b:
        v = members[:, :, j]
        med = np.nanmedian(v, axis=1)
        x_fine, y_fine = smooth_curve(n, med, transform=slog, inverse=inv_slog)
        ax_b.plot(x_fine, y_fine, color=color, lw=1.8, alpha=0.6, label=label, zorder=2)
        ax_b.scatter(n, med, color=color, s=20, alpha=0.6, zorder=3)
        n_repeated = np.repeat(n, v.shape[1])
        ax_b.scatter(n_repeated, v.ravel(), color=color, s=18, marker="x", lw=0.8, zorder=4)
    ax_b.axhline(0.0, color=style.GRID, lw=0.8, zorder=0)
    ax_b.axvline(horizon_days, color=style.INK_MUTED, lw=0.9, ls=(0, (1, 2)))
    ax_b.set_xscale("log")
    ax_b.set_yscale("symlog", linthresh=LINTHRESH)
    ax_b.set_yticks([sign * 10.0**k for sign in (1, -1) for k in (-1, 3, 7, 11)])
    ax_b.set_xlim(n.min() * 0.85, n.max() * 1.2)
    ax_b.set_xlabel("rollout length (days)")
    ax_b.set_ylabel(rf"$\partial L\,/\,\partial\,${SWARM_PARAM}")
    ax_b.set_title(f"(b) {SWARM_PARAM}: gradient value and round-off spread")
    ax_b.legend(loc="upper left", fontsize=8.5)

    for ax in (ax_a, ax_b):
        add_steps_axis(ax, dt)

    fig.suptitle("Gradient of mean-square surface temperature w.r.t. model parameters",
                 fontsize=10.5, y=1.1, color=style.INK)
    fig.tight_layout()
    style.save(fig, common.FIG_DIR / "fig1_gradient_check")


if __name__ == "__main__":
    main()
