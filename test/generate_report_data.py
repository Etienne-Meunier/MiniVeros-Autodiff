#!/usr/bin/env python3
"""
Long-rollout data generator for the trust report (see ../TRUST_REPORT.md).

Runs mini_veros vs real veros for each setup over a longer horizon than the
quick regression checks, recording:
  - error evolution (all prognostic fields, at record_interval steps)
  - full snapshots (mini, real, diff) of a few representative fields at the
    final step, for visualization

Saves one .npz per setup into test/results/. A separate plotting step
(plot_report_figures.py) reads these back in -- kept separate so re-plotting
doesn't require re-running the (slow) simulations.

Usage:
    python test/generate_report_data.py [--veros-path PATH]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from util import build_mini, build_real, configure_veros_runtime, compute_error_evolution

DEFAULT_VEROS_PATH = REPO_ROOT.parent / "veros"
REAL_SETUP_CLASS = {
    "acc_basic": "ACCBasicSetup",
    "acc": "ACCSetup",
    "global_4deg": "GlobalFourDegreeSetup",
}

# (n_steps, record_interval) -- global_4deg is much more expensive per step
# (larger grid + real climatology forcing), so it gets a shorter horizon.
RUN_CONFIG = {
    "acc_basic": (300, 10),
    "acc": (300, 10),
    "global_4deg": (300, 10),
}

RESULTS_DIR = REPO_ROOT / "test" / "results"


def run_setup(setup_name, veros_path):
    n_steps, record_interval = RUN_CONFIG[setup_name]
    print(f"--- {setup_name}: {n_steps} steps, recording every {record_interval} ---")

    t0 = time.time()
    _, mini_s0, mini_sf, timesteps, mini_states = build_mini(
        setup_name, n_steps, veros_path, record_interval
    )
    t1 = time.time()
    _, real_s0, real_sf, _, real_states = build_real(
        setup_name, n_steps, veros_path, REAL_SETUP_CLASS, record_interval
    )
    t2 = time.time()
    print(f"    mini: {t1 - t0:.1f}s   real: {t2 - t1:.1f}s")

    errors = compute_error_evolution(timesteps, mini_states, real_states)

    snapshots = {}
    for field in ("psi", "temp", "u", "salt"):
        if field not in mini_sf:
            continue
        snapshots[f"{field}_mini"] = mini_sf[field]
        snapshots[f"{field}_real"] = real_sf[field]

    out = dict(timesteps=np.asarray(timesteps))
    for field, data in errors.items():
        for key in ("max_abs_errors", "max_rel_errors", "mean_abs_errors", "median_abs_errors"):
            out[f"err_{field}_{key}"] = np.asarray(data[key])
    out.update(snapshots)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{setup_name}.npz"
    np.savez(out_path, **out)
    print(f"    saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--veros-path", type=Path, default=DEFAULT_VEROS_PATH)
    parser.add_argument("--setup", choices=sorted(RUN_CONFIG) + ["all"], default="all")
    args = parser.parse_args()

    if not (args.veros_path / "veros" / "__init__.py").exists():
        parser.error(f"no veros package found at {args.veros_path}")

    configure_veros_runtime(args.veros_path)

    setups = list(RUN_CONFIG) if args.setup == "all" else [args.setup]
    for setup_name in setups:
        run_setup(setup_name, args.veros_path)


if __name__ == "__main__":
    main()
