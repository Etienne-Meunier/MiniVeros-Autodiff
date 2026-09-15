"""
Figure 3 -- fitting two hidden mixing parameters.

Left to right: the target surface temperature, the surface error before the fit,
the same error after it (both on one shared colour scale, so the shrinkage is
the message), the loss landscape with the optimiser's path drawn on it, and the
loss per iteration.

    python test/paper_figures/fig3_calibration.py
"""

import matplotlib.pyplot as plt
import numpy as np

import common
import style


def main():
    style.use()
    d = common.load("exp3_calibration")
    x, y, land = d["x"], d["y"], d["land"]
    names = [str(p) for p in d["param_names"]]
    true, guess, fitted = d["true"], d["guess"], d["fitted"]
    traj, losses = d["trajectory"], d["losses"]

    before = d["sst_guess"] - d["sst_true"]
    after = d["sst_fitted"] - d["sst_true"]
    err_norm = style.symmetric(np.where(land, np.nan, before))
    rms = lambda f: np.sqrt(np.nanmean(np.where(land, np.nan, f) ** 2))

    fig = plt.figure(figsize=(14.5, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1.7, 0.95])
    ax_t, ax_b, ax_a, ax_l, ax_c = [fig.add_subplot(gs[0, i]) for i in range(5)]

    mesh = style.map_panel(ax_t, x, y, d["sst_true"], land, cmap=style.SEQ_WARM, title="(a) target SST")
    style.colorbar(fig, mesh, ax_t, "K")

    style.map_panel(ax_b, x, y, before, land, cmap=style.DIVERGING, norm=err_norm,
                    title=f"(b) before fit · rms {rms(before):.1e} K")
    mesh = style.map_panel(ax_a, x, y, after, land, cmap=style.DIVERGING, norm=err_norm,
                           title=f"(c) after fit · rms {rms(after):.1e} K")
    style.colorbar(fig, mesh, [ax_b, ax_a], "SST error (K)")
    for ax in (ax_t, ax_b, ax_a):
        ax.set_xlabel("longitude")
    ax_t.set_ylabel("latitude")
    ax_b.set_yticklabels([])
    ax_a.set_yticklabels([])

    # Landscape: log10 of the loss, so a valley several decades deep stays readable.
    floor = np.nanmin(d["landscape"][d["landscape"] > 0]) / 2
    logL = np.log10(np.maximum(d["landscape"], floor))
    mesh = ax_l.contourf(d["c_k"], d["c_eps"], logL.T, levels=24, cmap=style.SEQ_BLUE)
    ax_l.contour(d["c_k"], d["c_eps"], logL.T, levels=12, colors=style.SURFACE, linewidths=0.5, alpha=0.55)
    style.colorbar(fig, mesh, ax_l, r"$\log_{10}$ loss")

    ax_l.plot(traj[:, 0], traj[:, 1], color=style.SERIES[1], marker="o", ms=3.5, lw=1.6,
              markevery=max(1, len(traj) // 15), mec=style.SURFACE, mew=0.5,
              label="descent path", zorder=3)
    ax_l.plot(*guess, marker="o", ms=9, mfc="none", mec=style.SERIES[1], mew=1.8, ls="none",
              label="start", zorder=4)
    ax_l.plot(*true, marker="*", ms=16, color=style.INK, mec=style.SURFACE, mew=0.6, ls="none",
              label="truth", zorder=5)
    ax_l.set_xlabel(names[0])
    ax_l.set_ylabel(names[1])
    ax_l.set_title(f"(d) loss landscape · fitted ({fitted[0]:.3f}, {fitted[1]:.3f})")
    ax_l.grid(False)
    ax_l.legend(loc="upper right", labelcolor=style.INK_SOFT, frameon=True,
                facecolor=style.SURFACE, edgecolor="none", framealpha=0.88)

    ax_c.plot(losses, color=style.SERIES[1])
    ax_c.set_yscale("log")
    ax_c.set_xlabel("iteration")
    ax_c.set_ylabel("loss")
    ax_c.set_title("(e) convergence")

    fig.suptitle(f"Calibrating ({names[0]}, {names[1]}) against T,S on the top {int(d['n_layers'])} layers "
                 f"at the end of a {int(d['n_steps'])}-day rollout",
                 fontsize=11, color=style.INK)
    style.save(fig, common.FIG_DIR / "fig3_calibration")


if __name__ == "__main__":
    main()
