#!/usr/bin/env python3
"""
Test suite comparing mini_veros and veros prognostic state.

Runs both models on the same setup for N steps and compares all prognostic
variables (u, v, temp, salt, psi, tke, eke where applicable).

Usage:
    python test/test_prognostic.py [--setup acc_basic|acc] [--steps N] [--veros-path PATH] [--atol A] [--rtol R]
"""

import argparse
import sys
from pathlib import Path

from util import (
    build_mini,
    build_real,
    configure_veros_runtime,
    print_section,
    compute_error_evolution,
    print_error_evolution,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VEROS_PATH = REPO_ROOT.parent / "veros"

REAL_SETUP_CLASS = {
    "acc_basic": "ACCBasicSetup",
    "acc": "ACCSetup",
    "global_4deg": "GlobalFourDegreeSetup",
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--setup",
        choices=sorted(REAL_SETUP_CLASS),
        default="acc_basic",
        help="which setup to compare (default: acc_basic)",
    )
    parser.add_argument(
        "--steps", type=int, default=5, help="number of step()s to run (default: 5)"
    )
    parser.add_argument(
        "--veros-path",
        type=Path,
        default=DEFAULT_VEROS_PATH,
        help=f"path to the real veros repo (default: {DEFAULT_VEROS_PATH})",
    )
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument(
        "--record-interval",
        type=int,
        default=None,
        help="record state every N steps for error evolution plot (default: None)",
    )
    args = parser.parse_args()

    if not (args.veros_path / "veros" / "__init__.py").exists():
        parser.error(f"no veros package found at {args.veros_path} -- pass --veros-path")

    print(f"Running {args.setup} for {args.steps} step(s)...")
    print(f"  veros:     {args.veros_path}")
    if args.record_interval:
        print(f"  record interval: every {args.record_interval} step(s)")

    configure_veros_runtime(args.veros_path)

    # Build with optional error evolution tracking
    result_mini = build_mini(
        args.setup, args.steps, args.veros_path, args.record_interval
    )
    result_real = build_real(
        args.setup, args.steps, args.veros_path, REAL_SETUP_CLASS, args.record_interval
    )

    if args.record_interval:
        mini_model, mini_s0, mini_sf, mini_timesteps, mini_states = result_mini
        real_sim, real_s0, real_sf, real_timesteps, real_states = result_real
    else:
        mini_model, mini_s0, mini_sf = result_mini
        real_sim, real_s0, real_sf = result_real
        mini_timesteps = real_timesteps = None

    ok = True
    ok &= print_section(
        "initial state (t=0)", mini_s0, real_s0, args.atol, args.rtol
    )
    ok &= print_section(
        f"state after {args.steps} step(s)", mini_sf, real_sf, args.atol, args.rtol
    )

    # Print error evolution table if requested
    if args.record_interval and mini_timesteps is not None:
        errors = compute_error_evolution(
            mini_timesteps, mini_states, real_states, args.atol, args.rtol
        )
        print_error_evolution(errors, args.rtol)

    print()
    if ok:
        print(
            f"PASS -- mini_veros matches veros within atol={args.atol:.1e} rtol={args.rtol:.1e}"
        )
    else:
        print("FAIL -- see mismatches above")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
