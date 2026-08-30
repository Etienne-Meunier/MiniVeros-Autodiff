#!/usr/bin/env python3
"""
wandb-agent entrypoint for running the matrix in parallel (e.g. across g5k
nodes), instead of generate_matrix_data.py's sequential for-loop.

One agent process = one variant per pull from the sweep's grid queue, so
launching N agents on N g5k nodes runs N variants concurrently; the queue
self-balances the acc/global cost imbalance since idle agents just pull the
next remaining variant rather than working off a static split.

Setup:
    wandb sweep test/sweep/matrix.yaml            # prints SWEEP_ID
    wandb agent SWEEP_ID                          # launch via g5k's own job tool, one agent per node

veros path / step overrides come from env vars, not sweep params -- wandb
agent only forwards the swept `variant` param to this script's argv:
    VEROS_PATH, MATRIX_STEPS, MATRIX_RECORD_INTERVAL
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from generate_matrix_data import DEFAULT_VEROS_PATH, RUN_CONFIG, run_variant
from setups_matrix import FAMILIES, VARIANTS_BY_NAME
from util import configure_veros_runtime


def main():
    wandb.init()
    variant_name = wandb.config.variant
    variant = dict(VARIANTS_BY_NAME[variant_name])

    veros_path = Path(os.environ.get("VEROS_PATH", DEFAULT_VEROS_PATH))
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

    configure_veros_runtime(veros_path)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    try:
        run_variant(variant, veros_path, run_timestamp)
        wandb.log({"status": "ok"})
    except Exception as e:
        # same tolerance as generate_matrix_data.py's loop: acc_minimal etc.
        # are expected to diverge -- record it, don't fail the agent, so it
        # moves on to the next queued variant.
        print(f"    FAILED: {variant_name}: {e}")
        wandb.log({"status": "failed", "error": str(e)})


if __name__ == "__main__":
    main()
