"""
Run the c_k x c_eps sweep: 25 acc/full channel runs (5x5 grid, factor-16 range
around the defaults c_k=0.10, c_eps=0.70), each integrated N_YEARS=30 model-years
forward from a cold start, logging a horizontally-averaged potential density
profile every 30 days. Writes one .npz to DATA_DIR with every run's full logged
profile history plus the last-year-averaged final profile.

    python test/ck_ceps_sweep/run_sweep.py [--years N] [--k-grid v1 v2 ...] [--eps-grid v1 v2 ...]

Re-running overwrites the .npz; there is no per-run cache (each of the 25 runs
takes a few minutes on a GPU, see README.md).
"""

import argparse
import time

import numpy as np

import common


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=common.N_YEARS)
    p.add_argument("--k-grid", type=float, nargs="+", default=list(common.C_K_GRID))
    p.add_argument("--eps-grid", type=float, nargs="+", default=list(common.C_EPS_GRID))
    args = p.parse_args()

    n_steps = args.years * common.STEPS_PER_YEAR
    assert n_steps % common.LOG_EVERY == 0, "years*STEPS_PER_YEAR must be a multiple of LOG_EVERY"
    n_logs = n_steps // common.LOG_EVERY

    zt, log_select_fn, run_fn = common.build_sweep_reductor()
    nz = zt.shape[0]

    k_grid = np.asarray(args.k_grid)
    eps_grid = np.asarray(args.eps_grid)
    profiles = np.full((len(k_grid), len(eps_grid), n_logs, nz), np.nan)
    final_profile = np.full((len(k_grid), len(eps_grid), nz), np.nan)
    diverged = np.zeros((len(k_grid), len(eps_grid)), dtype=bool)

    t0 = time.time()
    total = len(k_grid) * len(eps_grid)
    done = 0
    for i, c_k in enumerate(k_grid):
        for j, c_eps in enumerate(eps_grid):
            t1 = time.time()
            prof, div = common.run_one(float(c_k), float(c_eps), log_select_fn, run_fn, n_steps=n_steps)
            done += 1
            dt = time.time() - t1
            if div or prof is None:
                diverged[i, j] = True
                print(f"[{done}/{total}] c_k={c_k:.4g} c_eps={c_eps:.4g}: DIVERGED ({dt:.1f}s)")
                continue
            prof = np.asarray(prof)
            profiles[i, j] = prof
            final_profile[i, j] = prof[-common.KEEP_LAST :].mean(axis=0)
            print(f"[{done}/{total}] c_k={c_k:.4g} c_eps={c_eps:.4g}: ok ({dt:.1f}s, "
                  f"total {(time.time() - t0) / 60:.1f} min elapsed)")

    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        common.SWEEP_NPZ,
        zt=np.asarray(zt), k_grid=k_grid, eps_grid=eps_grid,
        c_k_default=common.C_K_DEFAULT, c_eps_default=common.C_EPS_DEFAULT,
        n_years=args.years, steps_per_year=common.STEPS_PER_YEAR, log_every=common.LOG_EVERY,
        keep_last=common.KEEP_LAST,
        profiles=profiles, final_profile=final_profile, diverged=diverged,
    )
    print(f"wrote {common.SWEEP_NPZ} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
