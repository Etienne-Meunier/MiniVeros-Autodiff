"""
Figure 2 -- four sensitivity maps dT/dp as they develop along a rollout.

One row per row of exp2 (wind stress scale, kappaH_min, heat flux scale,
eke_c_k). Within a row: the surface sensitivity at each snapshot (one shared,
percentile-set colour scale per row, so the growth from column to column is
the message), then one more column for the sensitivity at ~1000 m at the last
snapshot, so a mixed-layer response and a thermocline response don't collide
on the same colour scale.

    python test/paper_figures/fig2_sensitivity_map.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import common
import style

ROW_TITLES = {
    "wind_stress_scale": r"wind stress scale $\alpha$",
    "kappaH_min": r"background diffusivity $\kappa_{H,\rm min}$",
    "heat_flux_scale": r"heat flux scale $\alpha$",
    "eke_c_k": r"EKE closure eke$\_$c$\_$k",
}


def centred(field, pct=99.5):
    v = float(np.nanpercentile(np.abs(field), pct))
    v = v if v > 0 else 1.0
    return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)


def main():
    style.use()
    d = common.load("exp2_sensitivity_map")
    rows = [str(r) for r in d["rows"]]
    x, y, z, land, dt = d["x"], d["y"], d["z"], d["land"], float(d["dt_tracer"])
    z_deep_idx = int(d["z_deep_idx"])
    n_snap = d[f"sensitivity__{rows[0]}"].shape[0]
    n_cols = n_snap + 1

    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.1 * n_cols + 1.4, 2.5 * len(rows)),
                              constrained_layout=True, squeeze=False)

    for r, name in enumerate(rows):
        sens = d[f"sensitivity__{name}"]  # (n_snap, nx, ny, nz)
        surf = sens[..., -1]
        norm = centred(np.where(land[None], np.nan, surf))

        for i in range(n_snap):
            days = int(d["steps"][i]) * dt / 86400
            peak = np.nanmax(np.abs(np.where(land, np.nan, surf[i])))
            mesh = style.map_panel(axes[r, i], x, y, surf[i], land, cmap=style.DIVERGING, norm=norm,
                                   title=f"{days:.0f} d\npeak {peak:.2g}" if r == 0 else f"peak {peak:.2g}")
            if r == len(rows) - 1:
                axes[r, i].set_xlabel("longitude")
            if i:
                axes[r, i].set_yticklabels([])

        deep_norm = centred(np.where(land, np.nan, sens[-1, ..., z_deep_idx]))
        mesh_deep = style.map_panel(axes[r, -1], x, y, sens[-1, ..., z_deep_idx], land,
                                    cmap=style.DIVERGING, norm=deep_norm,
                                    title=f"~{abs(float(z[z_deep_idx])):.0f} m" if r == 0 else None)
        axes[r, -1].set_yticklabels([])
        if r == len(rows) - 1:
            axes[r, -1].set_xlabel("longitude")

        axes[r, 0].set_ylabel(f"{ROW_TITLES.get(name, name)}\nlatitude", fontsize=8.5)
        style.colorbar(fig, mesh, list(axes[r, :-1]), rf"$\partial T_{{\rm surf}}/\partial p$")

    fig.suptitle("Sensitivity of temperature to four namelist quantities, propagated in forward mode",
                 fontsize=11, color=style.INK)
    style.save(fig, common.FIG_DIR / "fig2_sensitivity_map")


if __name__ == "__main__":
    main()
