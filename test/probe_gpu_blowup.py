#!/usr/bin/env python3
"""
Why does mini_veros go non-finite on GPU where it does not on CPU?

`acc_biharmonic_mixing` is the reproducer: the 20260902T220817Z (cpu) sweep
ran it to 10950 steps with both codes healthy, while 20260903T172502Z (gpu)
recorded mini_veros first non-finite at step 1800 with veros never tripping
its own sanity check. Same code, same tolerance, same horizon -- only the
jax platform differs.

The matrix records every 150 steps, which is far too coarse to see what
happens. This runs mini_veros alone at a fine interval and reports:

  - the first recorded step at which each field stops being finite, so the
    order tells us which term goes first (psi first would point at the
    elliptic solve, temp first at the biharmonic operator that feeds it)
  - max |field| over time, which separates a gradual instability (values
    climbing over hundreds of steps) from a sudden NaN (a divide or a sqrt
    of a negative, finite one step and NaN the next)
  - optionally the same run on the other device, diffed step by step, to
    locate the first step where the two platforms part company

Usage:
    python test/probe_gpu_blowup.py --device gpu
    python test/probe_gpu_blowup.py --device gpu --compare-cpu --steps 2000
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))


def run_mini(variant_name, veros_path, device, n_steps, record_interval, solver_atol):
    """One mini_veros run on `device`. Returns (timesteps, [state dicts])."""
    from util import configure_veros_runtime

    configure_veros_runtime(veros_path, device=device)
    sys.path.insert(0, str(REPO_ROOT / "mini-veros"))

    from setups_matrix import VARIANTS_BY_NAME
    from variant_util import build_mini_variant, forced_solver_atol

    variant = VARIANTS_BY_NAME[variant_name]
    with forced_solver_atol(solver_atol):
        _, _, _, _, timesteps, states = build_mini_variant(
            variant["name"], variant["family"], variant["overrides"],
            n_steps, veros_path, record_interval,
        )
    return timesteps, states


def describe(timesteps, states, label):
    """First non-finite step per field, plus the growth that led there."""
    fields = sorted(states[0])
    print(f"\n=== {label}: first non-finite step per field ===")
    first = {}
    for f in fields:
        for i, t in enumerate(timesteps):
            if not np.all(np.isfinite(np.asarray(states[i][f]))):
                first[f] = (int(t), i)
                break
    if not first:
        print("  every field finite for the whole run")
    for f in sorted(first, key=lambda f: first[f][0]):
        print(f"  {f:6s} at step {first[f][0]}")

    onset = min((v[1] for v in first.values()), default=len(timesteps))
    lo = max(0, onset - 8)
    print(f"\n=== {label}: max |field| approaching the blow-up ===")
    print("step  " + "".join(f"{f:>12s}" for f in fields))
    for i in range(lo, min(onset + 2, len(timesteps))):
        row = []
        for f in fields:
            a = np.asarray(states[i][f], dtype=np.float64)
            row.append(np.nanmax(np.abs(a)) if np.any(np.isfinite(a)) else np.nan)
        print(f"{timesteps[i]:5d} " + "".join(f"{x:12.3e}" for x in row))
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="acc_biharmonic_mixing")
    parser.add_argument("--device", default="gpu", choices=("cpu", "gpu"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--record-interval", type=int, default=10)
    parser.add_argument("--solver-atol", type=float, default=1e-14)
    parser.add_argument("--veros-path", type=Path, default=REPO_ROOT / "veros")
    parser.add_argument("--compare-cpu", action="store_true",
                        help="also run on cpu in a subprocess and report the first step where the "
                             "two platforms diverge (jax pins the platform per process)")
    args = parser.parse_args()

    timesteps, states = run_mini(
        args.variant, args.veros_path, args.device, args.steps, args.record_interval, args.solver_atol
    )
    describe(timesteps, states, f"{args.variant} on {args.device}")

    if args.compare_cpu:
        import json
        import subprocess
        import tempfile

        # a second process: jax fixes the platform at first use, so the other
        # device cannot be exercised in this one
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cpu.npz"
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--variant", args.variant,
                 "--device", "cpu", "--steps", str(args.steps),
                 "--record-interval", str(args.record_interval),
                 "--solver-atol", str(args.solver_atol), "--_dump", str(out)],
                check=True,
            )
            other = np.load(out, allow_pickle=True)
            meta = json.loads(str(other["meta"]))
            print(f"\n=== {args.device} vs cpu, step by step ===")
            print("step  " + "".join(f"{f:>12s}" for f in meta["fields"]))
            for i, t in enumerate(timesteps):
                if i >= len(other["timesteps"]):
                    break
                row = []
                for f in meta["fields"]:
                    a = np.asarray(states[i][f], np.float64)
                    b = other[f"f_{f}"][i]
                    scale = np.sqrt(np.nanmean(b**2)) or 1.0
                    row.append(np.nanmax(np.abs(a - b)) / scale)
                if i % 10 == 0 or any(not np.isfinite(x) or x > 1e-6 for x in row):
                    print(f"{t:5d} " + "".join(f"{x:12.3e}" for x in row))


if __name__ == "__main__":
    # internal: --_dump makes this a worker that saves its run for --compare-cpu
    if "--_dump" in sys.argv:
        idx = sys.argv.index("--_dump")
        dump_path = sys.argv[idx + 1]
        del sys.argv[idx : idx + 2]
        args, _ = argparse.ArgumentParser().parse_known_args()
        import json

        p = argparse.ArgumentParser()
        p.add_argument("--variant"); p.add_argument("--device"); p.add_argument("--steps", type=int)
        p.add_argument("--record-interval", type=int); p.add_argument("--solver-atol", type=float)
        a = p.parse_args()
        ts, st = run_mini(a.variant, REPO_ROOT / "veros", a.device, a.steps, a.record_interval, a.solver_atol)
        fields = sorted(st[0])
        np.savez(dump_path, timesteps=np.asarray(ts),
                 meta=np.asarray(json.dumps({"fields": fields})),
                 **{f"f_{f}": np.stack([s[f] for s in st]) for f in fields})
    else:
        main()
