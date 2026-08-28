#!/usr/bin/env python3
"""
Runs every variant in setups_matrix.py (mini_veros vs veros), recording:
  - error evolution over the run (all prognostic fields)
  - field snapshots at every recorded step, for the report's gifs
  - average wall time per step, for both implementations

Saves one .npz per variant into $STORE/MiniVeros-Autodiff/results/, named
"{variant}__{timestamp}.npz" -- every variant run in one invocation shares
the same timestamp. Re-running (e.g. just `--variant acc_basic`) adds a new
timestamped file alongside older ones rather than overwriting, so
plot_matrix_report.py can pick a specific snapshot with --timestamp, or
the newest one per variant with the default "latest". A separate plotting
step (plot_matrix_report.py) reads these back in -- kept separate so
re-plotting doesn't require re-running the (slow) simulations.

acc variants run a longer horizon than global ones -- global_4deg is much
more expensive per step (bigger grid + real climatology forcing).

Usage:
    python test/generate_matrix_data.py                    # every variant
    python test/generate_matrix_data.py --variant acc_basic
    python test/generate_matrix_data.py --group acc         # acc family only
    python test/generate_matrix_data.py --steps 4 --record-interval 2 --time-steps 2 --variant acc_basic   # fast smoke test
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from setups_matrix import FAMILIES, VARIANTS, VARIANTS_BY_NAME
from util import compute_error_evolution, configure_veros_runtime
from variant_util import build_mini_variant, build_real_variant

DEFAULT_VEROS_PATH = REPO_ROOT / "veros"
STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
RESULTS_DIR = STORE / "MiniVeros-Autodiff" / "results"

# (n_steps, record_interval, time_n_steps) per group -- acc is cheap enough
# for a long horizon; global_4deg's bigger grid + climatology forcing gets a
# shorter one so the whole matrix finishes in reasonable time.
RUN_CONFIG = {
    "acc": dict(n_steps=300, record_interval=10, time_n_steps=20),
    "global": dict(n_steps=60, record_interval=5, time_n_steps=10),
}

# Fields snapshotted at every recorded step for the report's gifs. "temp"
# (uppermost level) shows surface heat transport; "psi" (already 2D, the
# barotropic streamfunction) shows the large-scale circulation.
SNAPSHOT_FIELDS = ("temp", "psi")


def run_variant(variant, veros_path, run_timestamp):
    name, family, overrides = variant["name"], variant["family"], variant["overrides"]
    group = FAMILIES[family]["group"]
    cfg = variant.get("run_config", RUN_CONFIG[group])
    n_steps, record_interval, time_n_steps = cfg["n_steps"], cfg["record_interval"], cfg["time_n_steps"]

    print(f"--- {name} ({group}): {n_steps} steps, recording every {record_interval}, "
          f"timing {time_n_steps} steps ---")

    t0 = time.time()
    _, mini_s0, mini_sf, mini_sec, mini_ts, mini_states = build_mini_variant(
        name, family, overrides, n_steps, veros_path, record_interval, time_n_steps
    )
    t1 = time.time()
    _, real_s0, real_sf, real_sec, real_ts, real_states = build_real_variant(
        name, family, overrides, n_steps, veros_path, record_interval, time_n_steps
    )
    t2 = time.time()
    print(f"    mini: {t1 - t0:.1f}s ({mini_sec * 1000:.2f} ms/step)   "
          f"real: {t2 - t1:.1f}s ({real_sec * 1000:.2f} ms/step)")

    assert mini_ts == real_ts, f"{name}: mini/real recorded different timesteps: {mini_ts} vs {real_ts}"

    errors = compute_error_evolution(mini_ts, mini_states, real_states)

    out = dict(
        timesteps=np.asarray(mini_ts),
        mini_sec_per_step=np.asarray(mini_sec),
        real_sec_per_step=np.asarray(real_sec),
        family=np.asarray(family),
        group=np.asarray(group),
        overrides_json=np.asarray(json.dumps(overrides)),
        generated_at=np.asarray(run_timestamp),
        run_config_json=np.asarray(json.dumps(dict(n_steps=n_steps, record_interval=record_interval, time_n_steps=time_n_steps))),
    )
    for field, data in errors.items():
        for key in ("max_abs_errors", "max_rel_errors", "mean_abs_errors", "median_abs_errors"):
            out[f"err_{field}_{key}"] = np.asarray(data[key])
        # per-step pass/fail from compare_field's np.allclose(atol, rtol) -- the
        # same criterion test_matrix.py gates on, unlike a bare max_rel
        # threshold (which false-flags u/v: their relative error blows up on
        # near-zero values even when the absolute difference is solver noise).
        out[f"err_{field}_passes"] = np.asarray(data["passes"])

    for field in SNAPSHOT_FIELDS:
        if field not in mini_states[0]:
            continue
        out[f"{field}_mini_frames"] = np.stack([s[field] for s in mini_states])
        out[f"{field}_real_frames"] = np.stack([s[field] for s in real_states])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}__{run_timestamp}.npz"
    np.savez(out_path, **out)
    print(f"    saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS_BY_NAME), default=None, help="run a single variant")
    parser.add_argument("--group", choices=("acc", "global"), default=None, help="run only this group's variants")
    parser.add_argument("--veros-path", type=Path, default=DEFAULT_VEROS_PATH)
    parser.add_argument("--steps", type=int, default=None, help="override n_steps for every selected variant")
    parser.add_argument("--record-interval", type=int, default=None, help="override record_interval")
    parser.add_argument("--time-steps", type=int, default=None, help="override time_n_steps")
    args = parser.parse_args()

    if not (args.veros_path / "veros" / "__init__.py").exists():
        parser.error(f"no veros package found at {args.veros_path}")

    configure_veros_runtime(args.veros_path)

    if args.variant:
        selected = [VARIANTS_BY_NAME[args.variant]]
    elif args.group:
        selected = [v for v in VARIANTS if FAMILIES[v["family"]]["group"] == args.group]
    else:
        selected = VARIANTS

    # one timestamp for the whole invocation, so a full run's variants share
    # a snapshot; a partial rerun (--variant/--group) gets its own, newer one
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for variant in selected:
        if args.steps or args.record_interval or args.time_steps:
            group = FAMILIES[variant["family"]]["group"]
            base = dict(RUN_CONFIG[group])
            if args.steps:
                base["n_steps"] = args.steps
            if args.record_interval:
                base["record_interval"] = args.record_interval
            if args.time_steps:
                base["time_n_steps"] = args.time_steps
            variant = dict(variant, run_config=base)
        run_variant(variant, args.veros_path, run_timestamp)


if __name__ == "__main__":
    main()
