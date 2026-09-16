"""
Ranks the 10x10 (c_k, c_eps) grid in EXTERNAL_100Y_DIR by RMS deviation of
its final-model-year basin-mean density profile from the grid mean -- same
method used to pick the original TOP5 from the 25-point sweep. Prints the
sorted ranking; the top 5 become the new common.TOP5.

    python test/ck_ceps_sweep/rank_100y_grid.py
"""

import time

import numpy as np

import common
from mini_veros.core.density.get_rho import get_potential_rho
from mini_veros.setups.acc import full

C_K_GRID_100Y = (0.0125, 0.01984, 0.0315, 0.05, 0.07937, 0.126, 0.2, 0.3175, 0.504, 0.8)
C_EPS_GRID_100Y = (0.0875, 0.1389, 0.2205, 0.35, 0.5556, 0.8819, 1.4, 2.222, 3.528, 5.6)


def main():
    ref_model, _, _ = full.build({})
    area = np.asarray(ref_model.grid.area_t[2:-2, 2:-2])
    ocean_xy = np.asarray(ref_model.boundary_conditions.maskT[2:-2, 2:-2, :]).any(axis=2)
    area_weight = area * ocean_xy
    denom = area_weight.sum()
    eq_of_state_type = ref_model.config.eq_of_state_type

    profiles = {}
    t0 = time.time()
    n = len(C_K_GRID_100Y) * len(C_EPS_GRID_100Y)
    i = 0
    for c_k in C_K_GRID_100Y:
        for c_eps in C_EPS_GRID_100Y:
            i += 1
            path = common.EXTERNAL_100Y_DIR / f"ck{c_k:.4g}_eps{c_eps:.4g}.npz"
            t1 = time.time()
            d = np.load(path)
            temp = d["temp"][-common.KEEP_LAST:, 2:-2, 2:-2, :]
            salt = d["salt"][-common.KEEP_LAST:, 2:-2, 2:-2, :]
            rho = get_potential_rho(eq_of_state_type, salt, temp, 0.0)
            profile = (rho * area_weight[None, :, :, None]).sum(axis=(1, 2)) / denom
            profiles[(c_k, c_eps)] = profile.mean(axis=0)
            print(f"[{i}/{n}] c_k={c_k:.4g} c_eps={c_eps:.4g} ({time.time() - t1:.1f}s, "
                  f"total {(time.time() - t0) / 60:.1f} min)")

    mean_profile = np.mean(np.stack(list(profiles.values())), axis=0)
    rms = {k: float(np.sqrt(((v - mean_profile) ** 2).mean())) for k, v in profiles.items()}
    ranked = sorted(rms.items(), key=lambda kv: -kv[1])

    print("\nrank  c_k       c_eps     rms_from_grid_mean")
    for r, ((c_k, c_eps), val) in enumerate(ranked):
        print(f"{r:3d}   {c_k:.4g}   {c_eps:.4g}   {val:.5f}")

    print("\nTOP5 =", tuple((c_k, c_eps) for (c_k, c_eps), _ in ranked[:5]))


if __name__ == "__main__":
    main()
