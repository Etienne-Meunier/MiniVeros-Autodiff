"""
wandb-agent entrypoint for exp2 through exp9 and their b-variants (see test/sweep/paper.yaml).

One agent process = one experiment script per pull from the sweep's grid
queue, so `g5k agent <sweep> --n-agents 8` runs up to 8 of them concurrently.
Requires the spin-up and exp1_gradient_check to already have run (via
run_paper_oar.sh) and common.HORIZON_DAYS to be set, since every script here
sizes its rollout off it.

    wandb sweep test/sweep/paper.yaml       # prints SWEEP_ID
    wandb agent SWEEP_ID                    # one agent per node
"""

import runpy
import sys

import jax
import wandb

sys.path.insert(0, ".")  # so `import common` resolves when run from the repo root


def main():
    wandb.init()
    experiment = wandb.config.experiment
    device = wandb.config.get("device", "cpu")

    if device == "gpu":
        devices = jax.devices()
        if devices[0].platform != "gpu":
            raise RuntimeError(f"device=gpu requested but jax sees {devices[0].platform}")

    print(f"running {experiment} on {device}")
    # wandb invokes this script as `sweep_paper.py --device=gpu --experiment=...`; runpy
    # doesn't touch sys.argv, so without this the target script's own argparse would see
    # those same flags and reject them as unrecognized.
    sys.argv = [f"test/paper_figures/{experiment}.py"]
    try:
        runpy.run_path(f"test/paper_figures/{experiment}.py", run_name="__main__")
    except Exception as e:
        wandb.log({"status": "failed", "error": str(e)})
        raise
    else:
        wandb.log({"status": "ok"})


if __name__ == "__main__":
    main()
