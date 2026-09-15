"""
Figure 5 -- recovering an initial state from top-layer observations, and where that stops working.

Top row, part (a), identifiable: the true initial temperature, the error the
fit starts from, and the error left after it (a perturbation confined to the
observed layers, so this should nearly vanish), plus the convergence of the
observed loss and the whole-volume state error.

Bottom row, part (b), non-identifiable: the same fit run from K different
initial guesses, each wrong only *below* the observed layers -- invisible to
the observation operator by construction. All K should reach the same
observed loss; panel (f) is the point of the figure, the rms error by depth
for each of the K fits, spread apart below the observed layers and collapsed
together inside them.

    python test/paper_figures/fig5_assimilation.py
"""

import matplotlib.pyplot as plt
import numpy as np

import common
import style


def ocean_rms_by_depth(field, land):
    """rms over horizontal ocean points, one value per depth level. field: (nx, ny, nz)."""
    ocean = ~land
    return np.sqrt(np.array([np.mean(field[..., k][ocean] ** 2) for k in range(field.shape[-1])]))


def main():
    style.use()
    d = common.load("exp5_assimilation")
    x, y, z, land = d["x"], d["y"], d["z"], d["land"]
    n_layers = int(d["n_layers"])
    truth = d["truth"]
    before, after = d["start_a"] - truth, d["fitted_a"] - truth
    loss_a, state_err_a = d["history_a"][:, 0], d["history_a"][:, 1]

    ocean = lambda f: np.where(land[:, :, None], np.nan, f)
    rms = lambda f: np.sqrt(np.nanmean(ocean(f) ** 2))
    err_norm = style.symmetric(np.where(land, np.nan, before[..., -1]))

    fig = plt.figure(figsize=(13.5, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.35])
    maps = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_conv_a = fig.add_subplot(gs[0, 3])

    titles = ["(a) true initial temperature",
              f"(b) initial error · rms {rms(before):.2f} K",
              f"(c) error after fit · rms {rms(after):.2f} K"]
    for ax, field, cmap, norm, title in zip(
        maps, [truth, before, after], [style.SEQ_WARM, style.DIVERGING, style.DIVERGING],
        [None, err_norm, err_norm], titles,
    ):
        mesh = style.map_panel(ax, x, y, field[..., -1], land, cmap=cmap, norm=norm, title=title)
        ax.set_xlabel("longitude")
        if cmap is style.SEQ_WARM:
            style.colorbar(fig, mesh, ax, "K")
        elif ax is maps[2]:
            style.colorbar(fig, mesh, [maps[1], maps[2]], "temperature error (K)")
    maps[0].set_ylabel("latitude")
    for ax in maps[1:]:
        ax.set_yticklabels([])

    ax_conv_a.plot(np.arange(len(loss_a)), loss_a / loss_a[0], color=style.SERIES[0], label="observed loss (T,S)")
    ax_conv_a.plot(np.arange(len(state_err_a)), state_err_a / state_err_a[0], color=style.SERIES[1],
                   label="state error (rms, whole volume)")
    ax_conv_a.set_yscale("log")
    ax_conv_a.set_xlabel("iteration")
    ax_conv_a.set_ylabel("relative to iteration 0")
    ax_conv_a.set_title("(d) part (a) convergence")
    ax_conv_a.legend(loc="lower left", fontsize=8, frameon=True, facecolor=style.SURFACE, edgecolor="none",
                     framealpha=0.88)

    # Part (b): K fits from perturbations confined below the observed layers.
    starts_b, fitted_b, history_b = d["starts_b"], d["fitted_b"], d["history_b"]  # (K, ...)
    k_starts = fitted_b.shape[0]

    ax_loss_b = fig.add_subplot(gs[1, 0])
    for k in range(k_starts):
        ax_loss_b.plot(history_b[k, :, 0], color=style.SERIES[k % len(style.SERIES)], label=f"start {k}")
    ax_loss_b.set_yscale("log")
    ax_loss_b.set_xlabel("iteration")
    ax_loss_b.set_ylabel("observed loss (T,S)")
    ax_loss_b.set_title("(e) part (b): all K fits match the observations")
    ax_loss_b.legend(fontsize=7.5, ncols=2)

    ax_profile = fig.add_subplot(gs[1, 1:3])
    for k in range(k_starts):
        err_k = ocean_rms_by_depth(fitted_b[k] - truth, land)
        ax_profile.plot(err_k, z, color=style.SERIES[k % len(style.SERIES)], marker="o", ms=3, label=f"start {k}")
    ax_profile.axhspan(z[-n_layers], z.max(), color=style.INK_MUTED, alpha=0.12, lw=0)
    ax_profile.annotate("observed layers", (ax_profile.get_xlim()[1], z[-n_layers]), ha="right", va="bottom",
                        fontsize=8, color=style.INK_MUTED, xycoords=("axes fraction", "data"))
    ax_profile.set_xlabel("rms recovered $-$ truth (K), horizontal ocean mean")
    ax_profile.set_ylabel("depth (m)")
    ax_profile.set_title("(f) recovered state error by depth -- the null space")
    ax_profile.legend(fontsize=7.5)

    ax_spread = fig.add_subplot(gs[1, 3])
    spread = np.std(fitted_b, axis=0)  # (nx, ny, nz), std across the K fits
    spread_profile = ocean_rms_by_depth(spread, land)
    ax_spread.plot(spread_profile, z, color=style.INK, marker="o", ms=3)
    ax_spread.axhspan(z[-n_layers], z.max(), color=style.INK_MUTED, alpha=0.12, lw=0)
    ax_spread.set_xlabel("std across K fits (K)")
    ax_spread.set_ylabel("depth (m)")
    ax_spread.set_title("spread across fits")

    fig.suptitle(f"Assimilating the initial temperature field from T,S on the top {n_layers} layers, "
                 f"{int(d['n_steps'])}-day rollout", fontsize=11, color=style.INK, y=1.03)
    style.save(fig, common.FIG_DIR / "fig5_assimilation")


if __name__ == "__main__":
    main()
