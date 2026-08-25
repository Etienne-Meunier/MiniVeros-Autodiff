#!/usr/bin/env python3
"""
Quick acc-only comparison run -- same machinery as compare.py (imported
directly, not duplicated), just hardcoded to --setup acc with a fixed step
count so it can be run with no arguments for a fast sanity check.

Usage:
    python test/compare_acc.py [N_STEPS]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import DEFAULT_VEROS_PATH, build_mini, build_real, configure_real_veros_runtime, print_section

SETUP = "acc"
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ATOL = 1e-8
RTOL = 1e-8


def main():
    veros_path = DEFAULT_VEROS_PATH
    if not (veros_path / "veros" / "__init__.py").exists():
        sys.exit(f"no veros package found at {veros_path}")

    print(f"Running {SETUP} for {N_STEPS} step(s)...")
    print("  mini_veros:", Path(__file__).resolve().parents[1])
    print("  veros:     ", veros_path)

    configure_real_veros_runtime(veros_path)

    _, _, mini_s0, mini_sf = build_mini(SETUP, N_STEPS, veros_path)
    _, _, real_s0, real_sf = build_real(SETUP, veros_path, N_STEPS)

    ok = True
    ok &= print_section("initial state (t=0)", mini_s0, real_s0, ATOL, RTOL)
    ok &= print_section(f"state after {N_STEPS} step(s)", mini_sf, real_sf, ATOL, RTOL)

    print()
    print(f"PASS -- acc matches veros within atol={ATOL:.1e} rtol={RTOL:.1e}" if ok else "FAIL -- see mismatches above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
