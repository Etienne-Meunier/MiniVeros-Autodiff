#!/usr/bin/env python3
"""
Baseline for "how much do two real-veros runs of the *same* setup differ":
runs the same variant (setups_matrix.py) twice as two separate processes
(fresh interpreter each time -- rules out shared JIT-cache/state artifacts)
and diffs run A against run B with the exact same machinery
(util.compute_error_evolution) test_matrix.py uses for mini vs real.

This gives the actual noise floor of real veros against itself, instead of
test_matrix.py's assumed atol=1e-8/rtol=1e-6. If the noise floor is ~0
(expected: veros's step is a pure function of state + settings, no RNG, and
JAX on CPU is deterministic given a fixed thread count), any nonzero
mini/real diff at the same horizon is real divergence, not "solver noise" --
this is the check that TODO.md's "establish noise floor first" item asks
for, so its result can retire that assumption instead of guessing at it.

Usage:
    python test/measure_noise_floor.py --variant acc_basic
    python test/measure_noise_floor.py --variant global_default --steps 60 --record-interval 5
    python test/measure_noise_floor.py --variant acc_basic --threads 1   # pin XLA intra-op threads
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from setups_matrix import FAMILIES, VARIANTS_BY_NAME
from util import compute_error_evolution, configure_veros_runtime, print_error_evolution
from variant_util import build_real_variant

DEFAULT_VEROS_PATH = REPO_ROOT / "veros"
STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
RESULTS_DIR = STORE / "MiniVeros-Autodiff" / "results"


def _run_child(variant_name, n_steps, record_interval, veros_path, out_path):
    variant = VARIANTS_BY_NAME[variant_name]
    configure_veros_runtime(veros_path)
    _, _, _, _, timesteps, recorded_states = build_real_variant(
        variant["name"], variant["family"], variant["overrides"], n_steps, veros_path, record_interval
    )
    fields = sorted(recorded_states[0])
    out = dict(timesteps=np.asarray(timesteps))
    for field in fields:
        out[f"{field}_frames"] = np.stack([s[field] for s in recorded_states])
    np.savez(out_path, **out)


def _load_states(npz_path):
    data = np.load(npz_path)
    timesteps = data["timesteps"].tolist()
    fields = sorted(k.removesuffix("_frames") for k in data.files if k.endswith("_frames"))
    states = [{field: data[f"{field}_frames"][i] for field in fields} for i in range(len(timesteps))]
    return timesteps, states


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS_BY_NAME), default="acc_basic")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--record-interval", type=int, default=5)
    parser.add_argument("--veros-path", type=Path, default=DEFAULT_VEROS_PATH)
    parser.add_argument("--threads", type=int, default=None, help="pin XLA_FLAGS intra-op thread count for both runs")
    parser.add_argument("--child-run", nargs=2, metavar=("VARIANT", "OUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not (args.veros_path / "veros" / "__init__.py").exists():
        parser.error(f"no veros package found at {args.veros_path}")

    if args.child_run:
        variant_name, out_path = args.child_run
        _run_child(variant_name, args.steps, args.record_interval, args.veros_path, out_path)
        return

    env = dict(os.environ)
    if args.threads is not None:
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false"
        env["OMP_NUM_THREADS"] = str(args.threads)

    tmp_a = RESULTS_DIR / f"_noise_floor_{args.variant}_a.npz"
    tmp_b = RESULTS_DIR / f"_noise_floor_{args.variant}_b.npz"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for label, tmp in (("A", tmp_a), ("B", tmp_b)):
        print(f"--- run {label}: {args.variant}, {args.steps} steps ---")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--variant", args.variant, "--steps", str(args.steps),
             "--record-interval", str(args.record_interval), "--veros-path", str(args.veros_path),
             "--child-run", args.variant, str(tmp)],
            check=True, env=env,
        )

    timesteps_a, states_a = _load_states(tmp_a)
    timesteps_b, states_b = _load_states(tmp_b)
    assert timesteps_a == timesteps_b, f"recorded different timesteps: {timesteps_a} vs {timesteps_b}"

    errors = compute_error_evolution(timesteps_a, states_a, states_b)
    print_error_evolution(errors)

    out = dict(timesteps=np.asarray(timesteps_a), variant=np.asarray(args.variant))
    for field, data in errors.items():
        for key in ("max_abs_errors", "max_rel_errors", "mean_abs_errors", "median_abs_errors"):
            out[f"err_{field}_{key}"] = np.asarray(data[key])

    out_path = RESULTS_DIR / f"noise_floor_{args.variant}.npz"
    np.savez(out_path, **out)
    tmp_a.unlink()
    tmp_b.unlink()
    print(f"saved {out_path}")

    worst = max(
        (max(d["max_abs_errors"]) for d in errors.values() if d["max_abs_errors"]),
        default=0.0,
    )
    print(f"\nnoise floor (max abs diff, run A vs run B, across all fields/steps): {worst:.3e}")


if __name__ == "__main__":
    main()
