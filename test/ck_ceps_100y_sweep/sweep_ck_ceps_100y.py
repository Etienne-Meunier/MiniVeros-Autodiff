#!/usr/bin/env python3
"""
wandb-agent entrypoint for the 100-model-year c_k/c_eps full-state sweep (see
test/sweep/ck_ceps_100y.yaml). One agent process = one (c_k, c_eps) point per
pull from the sweep's grid queue, so `g5k agent <sweep> --n-agents 25` runs
the whole 5x5 grid concurrently, one GPU per point.

Mirrors test/sweep_matrix_data.py's pattern: the sweep id is the run id, so
every agent lands in results/ck_ceps_100y_sweep/full_state/{sweep_id}/ without
any shared timestamp to pass around. A divergent point is written as a
failure record rather than crashing the agent, so it moves on to the next
queued point.

Each point also writes a checkpoint (results/ck_ceps_100y_sweep/checkpoints/
{sweep_id}/...) -- the full IntegratorState at the final step, not just the
monthly PrognosticState snapshots -- so resume_point.py can continue it later
without re-running from year 0.

    wandb sweep test/sweep/ck_ceps_100y.yaml       # prints SWEEP_ID
    g5k agent <entity>/<project>/<sweep_id> --site grenoble --n-agents 25 \\
        --walltime 6:00:00 --no-besteffort
"""

import sys
from pathlib import Path

import jax
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import common` resolves regardless of cwd
import common


def _assert_gpu_available():
    """A GPU sweep that silently fell back to CPU is worse than a failed one: the
    run would be labelled gpu but take an order of magnitude longer than expected."""
    devices = jax.devices()
    print(f"    jax devices: {devices}")
    if devices[0].platform != "gpu":
        raise RuntimeError(f"device=gpu requested but jax sees {devices[0].platform}")


def main():
    wandb.init()
    c_k = float(wandb.config.c_k)
    c_eps = float(wandb.config.c_eps)
    device = wandb.config.get("device", "gpu")
    if device == "gpu":
        _assert_gpu_available()

    # The sweep id is the run id: one value per sweep, identical across agents
    # because wandb issued it (see test/sweep_matrix_data.py for the same choice).
    run_id = wandb.run.sweep_id
    if not run_id:
        run_id = f"local-{wandb.run.id}"
        print(f"    not running under a sweep; using run id {run_id}")
    path = common.point_path(run_id, c_k, c_eps)
    ckpt_path = common.checkpoint_path(run_id, c_k, c_eps)
    print(f"    run_id={run_id} c_k={c_k:.4g} c_eps={c_eps:.4g} n_steps={common.N_STEPS} -> {path}")

    try:
        zt, states, final_integrator_state = common.run_one_point(c_k, c_eps)
        common.save_point(path, c_k, c_eps, zt, states)
        common.save_checkpoint(ckpt_path, c_k, c_eps, zt, final_integrator_state)
        wandb.log({"status": "ok"})
    except Exception as e:
        print(f"    FAILED: c_k={c_k:.4g} c_eps={c_eps:.4g}: {type(e).__name__}: {e}")
        wandb.log({"status": "failed", "error": str(e)})


if __name__ == "__main__":
    main()
