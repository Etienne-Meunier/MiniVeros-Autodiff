import matplotlib.pyplot as plt
import numpy as np

import common
import style

style.use()
d = common.load("exp11_density_calibration_limit")
lengths = list(d["lengths"])
i = lengths.index(5)
c_k, c_eps, true = d["c_k"], d["c_eps"], d["true"]
log_ck, log_ceps = np.log(c_k), np.log(c_eps)
log_true = np.log(true)
loss = d["landscape"][i]

argmin_idx = np.argmin(loss, axis=1)
u_pts = log_ck - log_true[0]
v_pts = log_ceps[argmin_idx] - log_true[1]
weight = 1.0 / loss[np.arange(len(c_k)), argmin_idx]
A = np.stack([u_pts, u_pts**2], axis=1)
sw = np.sqrt(weight)
coeffs, *_ = np.linalg.lstsq(sw[:, None] * A, sw * v_pts, rcond=None)

u_line = np.linspace(-0.8, 0.8, 200)
v_line = coeffs[0] * u_line + coeffs[1] * u_line**2

floor = np.nanmin(loss[loss > 0]) / 2
logL = np.log10(np.maximum(loss, floor))

fig, ax = plt.subplots(figsize=(6.5, 5.5))
mesh = ax.contourf(c_k, c_eps, logL.T, levels=20, cmap="viridis")
ax.contour(c_k, c_eps, logL.T, levels=10, colors=style.SURFACE, linewidths=0.4, alpha=0.5)
style.colorbar(fig, mesh, ax, r"$\log_{10}$ loss")

# per-column argmin points, sized/coloured by their weight (deeper = bigger/brighter)
ax.scatter(c_k, np.exp(log_ceps[argmin_idx]), s=30 + 200 * weight / weight.max(),
           color="white", edgecolor="black", linewidth=0.6, zorder=4, label="per-c_k argmin")

ax.plot(np.exp(log_true[0] + u_line), np.exp(log_true[1] + v_line), color="red", lw=2,
        zorder=5, label="weighted quadratic fit (through truth)")
ax.plot(*true, marker="*", ms=16, color=style.INK, mec=style.SURFACE, mew=0.6, ls="none",
        zorder=6, label="truth")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(c_k.min(), c_k.max())
ax.set_ylim(c_eps.min(), c_eps.max())
style.natural_log_ticks(ax, [0.05, 0.075, 0.10, 0.15, 0.20, 0.25], [0.3, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("tke_closure.c_k")
ax.set_ylabel("tke_closure.c_eps")
ax.set_title("len=5d: per-c_k-column argmin + weighted quadratic fit through truth")
ax.grid(False)
ax.legend(loc="upper left", fontsize=8)

out = common.FIG_DIR.parent / "_scratch_ridge_poly.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
