#!/usr/bin/env python3
"""
Regression check: mini_veros vs real veros across every setup, asserting
errors stay at solver-noise level (no growth, no order-of-magnitude gaps).

Point-wise relative error is a bad metric for psi: psi has an arbitrary
per-island gauge, and near-zero-magnitude cells make rel-error blow up even
when the absolute difference is physically negligible. So for psi we check
max_abs error normalized against the field's OWN physical scale (max|psi|)
instead of point-wise relative error. Every other prognostic field uses
plain atol/rtol (they don't have this near-zero-gauge problem).

Usage:
    python test/test_all_setups.py [--steps N] [--veros-path PATH]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from util import build_mini, build_real, configure_veros_runtime, compare_field

DEFAULT_VEROS_PATH = REPO_ROOT.parent / "veros"
REAL_SETUP_CLASS = {
    "acc_basic": "ACCBasicSetup",
    "acc": "ACCSetup",
    "global_4deg": "GlobalFourDegreeSetup",
}

ATOL = 1e-8
RTOL = 1e-6
PSI_SCALE_TOL = 1e-6  # max_abs(psi diff) / max_abs(real psi)


def check_setup(setup_name, n_steps, veros_path):
    model, mini_s0, mini_sf = build_mini(setup_name, n_steps, veros_path)
    sim, real_s0, real_sf = build_real(setup_name, n_steps, veros_path, REAL_SETUP_CLASS)

    ok = True
    lines = []
    for label, mini_state, real_state in (("t=0", mini_s0, real_s0), (f"t={n_steps}", mini_sf, real_sf)):
        for name in sorted(mini_state.keys()):
            if name not in real_state:
                continue
            mini_arr = mini_state[name]
            real_arr = real_state[name]

            if name == "psi":
                scale = np.max(np.abs(real_arr))
                scale = scale if scale > 0 else 1.0
                normalized = np.max(np.abs(mini_arr - real_arr)) / scale
                passed = normalized < PSI_SCALE_TOL
                lines.append(f"  [{label}] {name:6s} scale-normalized error={normalized:.3e} {'ok' if passed else 'FAIL'}")
            else:
                r = compare_field(name, mini_arr, real_arr, atol=ATOL, rtol=RTOL)
                passed = r["ok"]
                lines.append(f"  [{label}] {name:6s} max_abs={r['max_abs']:.3e} max_rel={r['max_rel']:.3e} {'ok' if passed else 'FAIL'}")

            ok = ok and passed

    return ok, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--veros-path", type=Path, default=DEFAULT_VEROS_PATH)
    args = parser.parse_args()

    if not (args.veros_path / "veros" / "__init__.py").exists():
        parser.error(f"no veros package found at {args.veros_path}")

    configure_veros_runtime(args.veros_path)

    all_ok = True
    for setup_name in ("acc_basic", "acc", "global_4deg"):
        print(f"\n=== {setup_name} ({args.steps} steps) ===")
        ok, lines = check_setup(setup_name, args.steps, args.veros_path)
        print("\n".join(lines))
        print("PASS" if ok else "FAIL")
        all_ok = all_ok and ok

    print()
    print("ALL SETUPS PASS" if all_ok else "SOME SETUPS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
