#!/usr/bin/env python3
"""
wandb-agent entrypoint for running the matrix in parallel (e.g. across g5k
nodes), instead of generate_matrix_data.py's sequential for-loop.

One agent process = one variant per pull from the sweep's grid queue, so
launching N agents on N g5k nodes runs N variants concurrently. The queue
self-balances the acc/global/global_1deg cost imbalance: an agent that
finishes a cheap variant pulls the next remaining one, rather than sitting
idle at the end of a statically assigned chunk.

Setup:
    wandb sweep test/sweep/matrix.yaml            # prints SWEEP_ID
    wandb agent SWEEP_ID                          # one agent per node

**The sweep id is the run id.** Results are written as
"results/{sweep_id}/{variant}.npz", so every agent lands in one directory
without any shared timestamp to pass around, and the report is rendered with
`plot_matrix_report.py --run-id SWEEP_ID`. This is the whole reason there is
no run-timestamp plumbing here any more: wandb already issues exactly one
identifier per sweep, shared by construction.

`device` is a sweep parameter, so the device the matrix ran on is in the sweep
record rather than in some node's environment. The remaining knobs stay
environment-read, because they are per-launch overrides rather than properties
of the sweep:

    MATRIX_SOLVER_ATOL     elliptic solver bound forced on both codes
    MATRIX_STEPS           override the group's horizon
    MATRIX_RECORD_INTERVAL override the group's record interval
    MATRIX_STORE_ALL_FIELDS / MATRIX_SNAPSHOT_SURFACE_ONLY   set to 1
    VEROS_PATH             defaults to the repo's veros/
    MATRIX_DEVICE          fallback for a sweep whose config has no `device`
"""

import os
import sys
from pathlib import Path

import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from generate_matrix_data import (DEFAULT_VEROS_PATH, RUN_CONFIG, run_variant,
                                  validate_run_id, write_failure_record)
from setups_matrix import FAMILIES, VARIANTS_BY_NAME
from util import configure_veros_runtime
from variant_util import DEFAULT_SOLVER_ATOL


def _flag(name):
    return os.environ.get(name, "0") not in ("0", "", "false", "False")


def _assert_gpu_available():
    """A GPU sweep that silently fell back to CPU is worse than a failed one: the
    .npz records device=gpu and the timings are wrong by an order of magnitude."""
    import jax

    devices = jax.devices()
    print(f"    jax devices: {devices}")
    if devices[0].platform != "gpu":
        raise RuntimeError(f"device=gpu requested but jax sees {devices[0].platform}")


def main():
    wandb.init()
    variant_name = wandb.config.variant
    variant = dict(VARIANTS_BY_NAME[variant_name])

    veros_path = Path(os.environ.get("VEROS_PATH", DEFAULT_VEROS_PATH))
    device = wandb.config.get("device") or os.environ.get("MATRIX_DEVICE", "cpu")
    solver_atol = float(os.environ.get("MATRIX_SOLVER_ATOL", DEFAULT_SOLVER_ATOL))
    store_all_fields = _flag("MATRIX_STORE_ALL_FIELDS")
    snapshot_surface_only = _flag("MATRIX_SNAPSHOT_SURFACE_ONLY")

    steps = os.environ.get("MATRIX_STEPS")
    record_interval = os.environ.get("MATRIX_RECORD_INTERVAL")
    if steps or record_interval:
        group = FAMILIES[variant["family"]]["group"]
        base = dict(RUN_CONFIG[group])
        if steps:
            base["n_steps"] = int(steps)
        if record_interval:
            base["record_interval"] = int(record_interval)
        variant["run_config"] = base

    # The sweep id is the run id: one value per sweep, identical across agents
    # because wandb issued it, not because a launch command remembered to pass
    # the same string to each node.
    run_id = wandb.run.sweep_id
    if not run_id:
        # `python test/sweep_matrix_data.py` outside a sweep. Nothing shared to
        # join, so name it after this run and say so.
        run_id = f"local-{wandb.run.id}"
        print(f"    not running under a sweep; using run id {run_id}")
    validate_run_id(run_id)

    configure_veros_runtime(veros_path, device=device)
    if device == "gpu":
        _assert_gpu_available()
    wandb.config.update(
        dict(run_id=run_id, device=device, solver_atol=solver_atol), allow_val_change=True
    )
    print(f"    run_id={run_id} device={device} -> {run_id}/{variant_name}.npz")

    try:
        run_variant(variant, veros_path, run_id, store_all_fields, solver_atol,
                    device, snapshot_surface_only)
        wandb.log({"status": "ok"})
    except Exception as e:
        # An unstable variant is a result, not a missing row: leave an .npz
        # saying so, so the report shows a failed row rather than an empty one
        # and you can tell "crashed" from "never ran". Don't fail the agent
        # either -- it should move on to the next queued variant.
        print(f"    FAILED: {variant_name}: {type(e).__name__}: {e}")
        write_failure_record(variant, run_id, e, solver_atol)
        wandb.log({"status": "failed", "error": str(e)})


if __name__ == "__main__":
    main()
