#!/usr/bin/env python3
"""
Resume one (c_k, c_eps) point from a checkpoint written by save_checkpoint (the full
IntegratorState -- state + tendency_m1/m2 + statefuldiag_m1 -- not the lighter
PrognosticState-only monthly snapshots sweep_ck_ceps_100y.py also writes). Runs
`--extra-years` more model-years from there, logging the full state every
LOG_EVERY_DAYS the same way, and writes both a continuation full-state .npz and a
fresh checkpoint at the new end point -- so a further resume can chain off this one.

    python resume_point.py --run-id <sweep_id> --c-k 0.4 --c-eps 0.175 --extra-years 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import common` resolves regardless of cwd
import common


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True, help="sweep id the checkpoint was written under")
    p.add_argument("--c-k", type=float, required=True)
    p.add_argument("--c-eps", type=float, required=True)
    p.add_argument("--extra-years", type=float, required=True, help="model-years to run past the checkpoint")
    p.add_argument("--out-run-id", default=None,
                   help="run id for the continuation's output (default: '<run-id>_resumed')")
    args = p.parse_args()

    out_run_id = args.out_run_id or f"{args.run_id}_resumed"
    ckpt_in = common.checkpoint_path(args.run_id, args.c_k, args.c_eps)
    n_steps = common.steps_for_years(args.extra_years)

    print(f"resuming c_k={args.c_k:.4g} c_eps={args.c_eps:.4g} from {ckpt_in}, "
          f"{n_steps} more steps (~{n_steps / common.STEPS_PER_YEAR:.2f} model-years)")
    zt, states, final_integrator_state = common.resume_one_point(
        args.c_k, args.c_eps, ckpt_in, args.extra_years
    )

    out_path = common.point_path(out_run_id, args.c_k, args.c_eps)
    ckpt_out = common.checkpoint_path(out_run_id, args.c_k, args.c_eps)
    common.save_point(out_path, args.c_k, args.c_eps, zt, states)
    common.save_checkpoint(ckpt_out, args.c_k, args.c_eps, zt, final_integrator_state)
    print(f"wrote {out_path}")
    print(f"wrote {ckpt_out}")


if __name__ == "__main__":
    main()
