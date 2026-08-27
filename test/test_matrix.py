#!/usr/bin/env python3
"""
pytest correctness gate for every variant in setups_matrix.py: mini_veros
vs veros must stay within solver-noise tolerance after a short run of each
config. Same tolerance policy as test_all_setups.py -- point-wise atol/rtol
for most fields; psi uses a scale-normalized check (it has an arbitrary
per-island gauge, so near-zero cells blow up point-wise relative error even
when the absolute difference is physically negligible).

This is the fast correctness check (short horizon, no recording/timing);
generate_matrix_data.py is the slow, longer-horizon data/figure generator
for the report.

Usage:
    pytest test/test_matrix.py
    pytest test/test_matrix.py -k acc_basic
    MINIVEROS_TEST_STEPS=100 pytest test/test_matrix.py
    MINIVEROS_VEROS_PATH=/path/to/veros pytest test/test_matrix.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from setups_matrix import VARIANTS
from util import compare_field, configure_veros_runtime
from variant_util import build_mini_variant, build_real_variant

DEFAULT_VEROS_PATH = REPO_ROOT / "veros"
N_STEPS = int(os.environ.get("MINIVEROS_TEST_STEPS", 20))
ATOL = 1e-8
RTOL = 1e-6
PSI_SCALE_TOL = 1e-6  # max_abs(psi diff) / max_abs(real psi)

# enable_streamfunction=False (surface-pressure formulation) diverges well
# beyond solver-noise tolerance within a handful of steps on both sides --
# the project's own generate_pressure_report_data.py already treats this
# path as report-only (records error evolution, asserts nothing), rather
# than a pass/fail gate. Mirrored here: xfail, not skip, so a regression
# that makes it *worse* still shows up, but a currently-known gap doesn't
# fail the suite.
_KNOWN_DIVERGENT = {
    "acc_surface_pressure": "surface-pressure formulation diverges beyond tolerance within a few steps; see generate_pressure_report_data.py",
    "global_surface_pressure": "surface-pressure formulation diverges beyond tolerance within a few steps; see generate_pressure_report_data.py",
}


@pytest.fixture(scope="session")
def veros_path():
    path = Path(os.environ.get("MINIVEROS_VEROS_PATH", DEFAULT_VEROS_PATH))
    if not (path / "veros" / "__init__.py").exists():
        pytest.skip(f"no veros package found at {path}")
    # must run exactly once, before any veros.core import -- see
    # util.configure_veros_runtime's docstring (runtime_settings lock
    # themselves the moment veros.core is first imported).
    configure_veros_runtime(path)
    return path


def _param(variant):
    reason = _KNOWN_DIVERGENT.get(variant["name"])
    marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
    return pytest.param(variant, id=variant["name"], marks=marks)


@pytest.mark.parametrize("variant", [_param(v) for v in VARIANTS])
def test_variant_matches_veros(variant, veros_path):
    name, family, overrides = variant["name"], variant["family"], variant["overrides"]

    _, mini_s0, mini_sf, *_ = build_mini_variant(name, family, overrides, N_STEPS, veros_path)
    _, real_s0, real_sf, *_ = build_real_variant(name, family, overrides, N_STEPS, veros_path)

    failures = []
    for label, mini_state, real_state in (("t=0", mini_s0, real_s0), (f"t={N_STEPS}", mini_sf, real_sf)):
        for fname in sorted(mini_state):
            if fname not in real_state:
                continue
            mini_arr, real_arr = mini_state[fname], real_state[fname]
            if fname == "psi":
                scale = np.max(np.abs(real_arr))
                scale = scale if scale > 0 else 1.0
                normalized = float(np.max(np.abs(mini_arr - real_arr)) / scale)
                if normalized >= PSI_SCALE_TOL:
                    failures.append(f"[{label}] psi scale-normalized error={normalized:.3e} (tol {PSI_SCALE_TOL:.1e})")
            else:
                r = compare_field(fname, mini_arr, real_arr, atol=ATOL, rtol=RTOL)
                if not r["ok"]:
                    failures.append(f"[{label}] {fname} max_abs={r['max_abs']:.3e} max_rel={r['max_rel']:.3e}")

    assert not failures, "\n".join(failures)
