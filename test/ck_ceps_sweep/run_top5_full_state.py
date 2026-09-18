"""
Reruns the 5 (c_k, c_eps) points with the largest deviation from the sweep's
grid-mean density profile (common.TOP5), this time logging the full
PrognosticState (u, v, temp, salt, tke, eke, psi) every 30 days instead of just
the reduced density profile -- ~1.1 MiB/snapshot x 365 snapshots x 5 runs
~= 2 GB total (float64). Writes one .npz per point to
DATA_DIR/full_state/ck{c_k}_eps{c_eps}.npz.

    python test/ck_ceps_sweep/run_top5_full_state.py [--years N]
"""

import argparse
import time

import equinox as eqx
import numpy as np

import common
from mini_veros import loop
from mini_veros.setups.acc import full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=common.N_YEARS)
    args = p.parse_args()

    n_steps = args.years * common.STEPS_PER_YEAR
    assert n_steps % common.LOG_EVERY == 0, "years*STEPS_PER_YEAR must be a multiple of LOG_EVERY"

    out_dir = common.DATA_DIR / "full_state"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_fn = eqx.filter_jit(loop.run)  # one JIT trace, reused for all 5 points

    t0 = time.time()
    for n, (c_k, c_eps) in enumerate(common.TOP5):
        model, state0, forcing_fn = full.build({"tke_closure.c_k": c_k, "tke_closure.c_eps": c_eps})
        t1 = time.time()
        try:
            _, states = run_fn(model, state0, forcing_fn, common.full_state_log_select_fn, n_steps, common.LOG_EVERY)
        except eqx.EquinoxRuntimeError as e:
            print(f"[{n + 1}/5] c_k={c_k:.4g} c_eps={c_eps:.4g}: DIVERGED ({e})")
            continue

        # states is a PrognosticState whose leaves each have a leading (n_logs,) axis.
        fields = {name: np.asarray(getattr(states, name))
                  for name in ("u", "v", "temp", "salt", "psi", "tke", "eke")
                  if getattr(states, name) is not None}

        path = out_dir / f"ck{c_k:.4g}_eps{c_eps:.4g}.npz"
        np.savez(path, c_k=c_k, c_eps=c_eps, zt=np.asarray(model.grid.zt),
                 n_years=args.years, log_every_days=common.LOG_EVERY_DAYS, **fields)
        print(f"[{n + 1}/5] c_k={c_k:.4g} c_eps={c_eps:.4g}: ok ({time.time() - t1:.1f}s) -> {path}")

    print(f"done ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
