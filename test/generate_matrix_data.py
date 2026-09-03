#!/usr/bin/env python3
"""
Runs every variant in setups_matrix.py (mini_veros vs veros), recording:
  - error evolution over the run (all prognostic fields), under the metrics
    in test/metrics.py: scale-normalized max, relative L2, pattern
    correlation, agreement horizon, and the climatology comparison
  - the legacy util.compute_error_evolution metrics, so older readers of
    these .npz files keep working
  - field snapshots at every recorded step, for the report's gifs -- temp and
    psi only, unless --store-all-fields
  - average wall time per step, for both implementations
  - solver_atol: the elliptic-solver stopping rule both codes were forced to
    (default 1e-14, tighter than the 1e-8 both ship with -- see --solver-atol)
  - a status: "ok", "diverged" (one side blew up mid-run -- the valid prefix
    is kept and compared), or "error" (the variant could not be run at all)

Every variant always leaves an .npz behind, including a failing one. It used
to leave nothing, and plot_matrix_report.py's "latest" resolution then fell
back to that variant's newest older file -- a 4-step smoke run, which passed
the tolerance gate only because 4 steps is not enough time to diverge.

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
    python test/generate_matrix_data.py --steps 4 --record-interval 2 --variant acc_basic   # fast smoke test
    python test/generate_matrix_data.py --store-all-fields   # self-contained .npz, ~3.5x the size
    python test/generate_matrix_data.py --solver-atol 1e-8   # the tolerance both codes ship with
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

import metrics
from setups_matrix import FAMILIES, VARIANTS, VARIANTS_BY_NAME
from util import compute_error_evolution, configure_veros_runtime
from variant_util import TIGHT_SOLVER_ATOL, build_mini_variant, build_real_variant, forced_solver_atol

DEFAULT_VEROS_PATH = REPO_ROOT / "veros"
STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
RESULTS_DIR = STORE / "MiniVeros-Autodiff" / "results"

# (n_steps, record_interval) per group -- acc is cheap enough for a long
# horizon; global_4deg's bigger grid + climatology forcing gets a shorter one
# so the whole matrix finishes in reasonable time. record_interval must divide
# n_steps (variant_util._run_steps runs the whole thing as one compiled scan);
# 150 is the nearest clean divisor of 365*30 to the old 100. sec_per_step is
# always measured as elapsed / n_steps over this same run -- no separate
# timing pass.
RUN_CONFIG = {
    "acc": dict(n_steps=365*30, record_interval=150),
    "global": dict(n_steps=365*30, record_interval=150),
}

# Fields snapshotted at every recorded step for the report's gifs. "temp"
# (uppermost level) shows surface heat transport; "psi" (already 2D, the
# barotropic streamfunction) shows the large-scale circulation.
#
# These two are also the only fields whose raw values survive the run: every
# metric is reduced in-process and only the reduction is written, so a new or
# corrected metric can be recomputed offline for temp/psi but needs a full
# rerun for u/v/salt/tke/eke. --store-all-fields keeps every prognostic
# field's frames instead, which makes the .npz self-contained at roughly 3.5x
# the size (~30 MB -> ~105 MB per acc variant).
SNAPSHOT_FIELDS = ("temp", "psi")


def _ms(sec_per_step):
    return "n/a" if sec_per_step is None else f"{sec_per_step * 1000:.2f} ms/step"


def write_failure_record(variant, run_timestamp, exc, solver_atol=TIGHT_SOLVER_ATOL):
    """
    Write a variant's .npz with status="error" and nothing else.

    The point is that the row exists: plot_matrix_report.py renders it as a
    failure with the message, instead of quietly picking up an older,
    shorter run of the same variant and reporting that as a pass.
    """
    name, family = variant["name"], variant["family"]
    group = FAMILIES[family]["group"]
    cfg = variant.get("run_config", RUN_CONFIG[group])
    out = dict(
        timesteps=np.asarray([], dtype=int),
        mini_sec_per_step=np.asarray(np.nan),
        real_sec_per_step=np.asarray(np.nan),
        family=np.asarray(family),
        group=np.asarray(group),
        overrides_json=np.asarray(json.dumps(variant["overrides"])),
        generated_at=np.asarray(run_timestamp),
        run_config_json=np.asarray(json.dumps(dict(n_steps=cfg["n_steps"], record_interval=cfg["record_interval"]))),
        status=np.asarray("error"),
        error_message=np.asarray(f"{type(exc).__name__}: {exc}"),
        real_diverged_at=np.asarray(-1),
        mini_nonfinite_at=np.asarray(-1),
        steps_completed=np.asarray(0),
        solver_atol=np.asarray(solver_atol),
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}__{run_timestamp}.npz"
    np.savez(out_path, **out)
    print(f"    recorded failure in {out_path}")


def run_variant(variant, veros_path, run_timestamp, store_all_fields=False, solver_atol=TIGHT_SOLVER_ATOL):
    name, family, overrides = variant["name"], variant["family"], variant["overrides"]
    group = FAMILIES[family]["group"]
    cfg = variant.get("run_config", RUN_CONFIG[group])
    n_steps, record_interval = cfg["n_steps"], cfg["record_interval"]

    print(f"--- {name} ({group}): {n_steps} steps, recording every {record_interval}, "
          f"solver atol {solver_atol:g} ---")

    # The patch has to span each build call in full, not just the stepping:
    # veros captures bicgstab in a closure during sim.setup(), and mini_veros
    # resolves it at trace time inside the jitted scan. See forced_solver_atol.
    with forced_solver_atol(solver_atol):
        t0 = time.time()
        _, mini_s0, mini_sf, mini_sec, mini_ts, mini_states = build_mini_variant(
            name, family, overrides, n_steps, veros_path, record_interval
        )
        t1 = time.time()
        # truncate rather than raise: an unstable variant is a result, not a
        # missing row. Real veros checks itself every step (numerics.sanity_check
        # in VerosSetup.step) and raises "solution diverged at iteration N"; that
        # used to abort the variant and leave the report silently falling back to
        # an older, much shorter snapshot of it.
        sim, real_s0, real_sf, real_sec, real_ts, real_states = build_real_variant(
            name, family, overrides, n_steps, veros_path, record_interval, stop_on_divergence=True
        )
        t2 = time.time()
    print(f"    mini: {t1 - t0:.1f}s ({_ms(mini_sec)})   real: {t2 - t1:.1f}s ({_ms(real_sec)})")

    real_diverged_at = getattr(sim, "diverged_at", None)
    mini_nonfinite = metrics.first_nonfinite(mini_ts, mini_states)

    if real_diverged_at is not None:
        print(f"    veros diverged at step {real_diverged_at}; keeping the {len(real_states)} valid records")
    if mini_nonfinite is not None:
        print(f"    mini_veros first non-finite at recorded step {mini_nonfinite[0]} ({mini_nonfinite[1]}); "
              f"comparison stops before it")

    # both sides record on the same schedule, so a truncated real run is a
    # prefix of the mini one -- compare over the common part
    n_common = min(len(mini_ts), len(real_ts))
    assert mini_ts[:n_common] == real_ts[:n_common], (
        f"{name}: mini/real recorded different timesteps: {mini_ts[:n_common]} vs {real_ts[:n_common]}"
    )

    # ...and stop before mini's first all-NaN record. Differencing NaN against
    # a number is not a comparison, and util.compare_field's np.nanargmax
    # raises "All-NaN slice encountered" on such a record rather than
    # returning anything usable.
    if mini_nonfinite is not None and mini_nonfinite[0] in mini_ts[:n_common]:
        n_common = min(n_common, mini_ts.index(mini_nonfinite[0]))
        n_common = max(n_common, 1)  # always keep step 0, which is exact by construction

    timesteps = mini_ts[:n_common]
    mini_states, real_states = mini_states[:n_common], real_states[:n_common]

    status = "ok"
    if real_diverged_at is not None or mini_nonfinite is not None:
        status = "diverged"

    out = dict(
        timesteps=np.asarray(timesteps),
        mini_sec_per_step=np.asarray(mini_sec if mini_sec is not None else np.nan),
        real_sec_per_step=np.asarray(real_sec if real_sec is not None else np.nan),
        family=np.asarray(family),
        group=np.asarray(group),
        overrides_json=np.asarray(json.dumps(overrides)),
        generated_at=np.asarray(run_timestamp),
        run_config_json=np.asarray(json.dumps(dict(n_steps=n_steps, record_interval=record_interval))),
        status=np.asarray(status),
        error_message=np.asarray(""),
        real_diverged_at=np.asarray(-1 if real_diverged_at is None else real_diverged_at),
        mini_nonfinite_at=np.asarray(-1 if mini_nonfinite is None else mini_nonfinite[0]),
        steps_completed=np.asarray(timesteps[-1] if timesteps else 0),
        solver_atol=np.asarray(solver_atol),
    )

    # legacy metrics, kept so older readers of these .npz files keep working
    errors = compute_error_evolution(timesteps, mini_states, real_states)
    for field, data in errors.items():
        for key in ("max_abs_errors", "max_rel_errors", "mean_abs_errors", "median_abs_errors"):
            out[f"err_{field}_{key}"] = np.asarray(data[key])
        out[f"err_{field}_passes"] = np.asarray(data["passes"])

    # the metrics the report actually reads (see test/metrics.py for why
    # max_rel is not among them)
    evolution = metrics.evolution(timesteps, mini_states, real_states)
    for field, per_metric in evolution.items():
        for metric_name, values in per_metric.items():
            out[f"m_{field}_{metric_name}"] = values
        step, exceeded = metrics.agreement_horizon(timesteps, per_metric["max_norm"])
        out[f"m_{field}_agreement_horizon"] = np.asarray(step)
        out[f"m_{field}_agreement_exceeded"] = np.asarray(exceeded)

        clim = metrics.climatology(
            [s[field] for s in mini_states], [s[field] for s in real_states]
        )
        if clim is not None:
            for key, value in clim.items():
                out[f"c_{field}_{key}"] = np.asarray(value)

    available = sorted(mini_states[0]) if mini_states else []
    wanted = available if store_all_fields else [f for f in SNAPSHOT_FIELDS if f in available]
    for field in wanted:
        out[f"{field}_mini_frames"] = np.stack([s[field] for s in mini_states])
        out[f"{field}_real_frames"] = np.stack([s[field] for s in real_states])
    # so a reader can tell whether a metric is recomputable from this file
    out["stored_fields"] = np.asarray(wanted)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}__{run_timestamp}.npz"
    np.savez(out_path, **out)
    print(f"    saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS_BY_NAME), default=None, help="run a single variant")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS_BY_NAME), default=None,
                        help="run this explicit list of variants (for splitting one sweep across jobs)")
    parser.add_argument("--group", choices=("acc", "global"), default=None, help="run only this group's variants")
    parser.add_argument("--veros-path", type=Path, default=DEFAULT_VEROS_PATH)
    parser.add_argument("--steps", type=int, default=None, help="override n_steps for every selected variant")
    parser.add_argument("--record-interval", type=int, default=None, help="override record_interval")
    parser.add_argument("--store-all-fields", action="store_true",
                        help="store every prognostic field's frames, not just temp/psi. Metrics are "
                             "reduced in-process, so by default a new or corrected metric can only be "
                             "recomputed offline for temp/psi and needs a full rerun for the rest; this "
                             "makes the .npz self-contained, at roughly 3.5x the size.")
    parser.add_argument("--solver-atol", type=float, default=TIGHT_SOLVER_ATOL,
                        help="absolute residual bound forced on BOTH codes' bicgstab for the external "
                             f"mode (default {TIGHT_SOLVER_ATOL:g}). Both ship with 1e-8, which is loose "
                             "enough that the two solvers stop ~1e-9 apart in relative psi -- seven "
                             "orders above float64 roundoff, and the largest avoidable seed of their "
                             "divergence. Costs 1-3%% in wall time. Pass 1e-8 to reproduce the shipped "
                             "configuration.")
    parser.add_argument("--run-timestamp", default=None,
                        help="stamp results with this instead of the current time. Pass the same value to "
                             "every job of a split sweep so the whole matrix lands on one snapshot -- "
                             "plot_matrix_report.py --strict then has a single timestamp to render.")
    args = parser.parse_args()

    if not (args.veros_path / "veros" / "__init__.py").exists():
        parser.error(f"no veros package found at {args.veros_path}")

    configure_veros_runtime(args.veros_path)

    if args.variant:
        selected = [VARIANTS_BY_NAME[args.variant]]
    elif args.variants:
        selected = [VARIANTS_BY_NAME[v] for v in args.variants]
    elif args.group:
        selected = [v for v in VARIANTS if FAMILIES[v["family"]]["group"] == args.group]
    else:
        selected = VARIANTS

    # one timestamp for the whole invocation, so a full run's variants share
    # a snapshot; a partial rerun (--variant/--group) gets its own, newer one
    # unless --run-timestamp pins it (split sweeps, see the flag's help)
    run_timestamp = args.run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for variant in selected:
        if args.steps or args.record_interval:
            group = FAMILIES[variant["family"]]["group"]
            base = dict(RUN_CONFIG[group])
            if args.steps:
                base["n_steps"] = args.steps
            if args.record_interval:
                base["record_interval"] = args.record_interval
            variant = dict(variant, run_config=base)
        try:
            run_variant(variant, args.veros_path, run_timestamp, args.store_all_fields, args.solver_atol)
        except Exception as e:
            # Don't let one variant's failure abort the rest of the matrix --
            # but do leave a file behind saying so. Printing only (the old
            # behaviour) wrote no .npz, and plot_matrix_report.py's "latest"
            # resolution then silently fell back to that variant's newest
            # older snapshot: a 4-step smoke run, which passed the tolerance
            # gate purely because 4 steps is not enough time to diverge.
            print(f"    FAILED: {variant['name']}: {type(e).__name__}: {e}")
            traceback.print_exc()
            write_failure_record(variant, run_timestamp, e, args.solver_atol)


if __name__ == "__main__":
    main()
