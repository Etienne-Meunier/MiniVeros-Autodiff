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

Every variant always leaves an .npz behind, including a failing one, so that
a variant which crashed is a visible failed row rather than a gap.

Saves one .npz per variant into $STORE/MiniVeros-Autodiff/results/{run_id}/,
named "{variant}.npz" -- every variant run in one invocation shares one run
directory, and re-running under a new id adds a directory alongside rather
than overwriting.

The run id is the addressing scheme -- and the directory name, so
`g5k sync model <run_id>` pulls exactly one run off the cluster.
plot_matrix_report.py renders exactly one id and nothing else. There is deliberately no "newest file wins"
resolution anywhere, because that is what let a variant missing from the
current run get backfilled from an older, shorter one -- a 4-step smoke run
passing the tolerance gate only because 4 steps is not enough time to
diverge.

Under a wandb sweep the id is the sweep id, so agents on different nodes land
in one addressable set without coordinating. Outside a sweep, --run-id names
it (default "local-<UTC timestamp>"). `generated_at` is still recorded inside
each .npz, but as provenance for the report banner only -- it orders nothing.

A separate plotting step (plot_matrix_report.py) reads these back in -- kept
separate so re-plotting doesn't require re-running the (slow) simulations.

acc variants run a longer horizon than global ones -- global_4deg is much
more expensive per step (bigger grid + real climatology forcing).

Usage:
    python test/generate_matrix_data.py                    # every variant
    python test/generate_matrix_data.py --variant acc_basic
    python test/generate_matrix_data.py --group acc         # acc family only
    python test/generate_matrix_data.py --steps 4 --record-interval 2 --variant acc_basic   # fast smoke test
    python test/generate_matrix_data.py --store-all-fields   # self-contained .npz, ~3.5x the size
    python test/generate_matrix_data.py --solver-atol 1e-8   # the tolerance both codes ship with
    python test/generate_matrix_data.py --variant global_1deg --run-id be7b0pze  # join an existing set
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
from run_ids import result_path, validate_run_id
from setups_matrix import FAMILIES, VARIANTS, VARIANTS_BY_NAME
from util import compute_error_evolution, configure_veros_runtime
from variant_util import DEFAULT_SOLVER_ATOL, build_mini_variant, build_real_variant, forced_solver_atol

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
    # 1-degree is ~56x global_4deg per step, so the 30-year horizon would be
    # tens of hours per variant. A short run still answers what the metrics
    # are sharpest at -- step-0 parity, physics parity, agreement horizon.
    # The climatology comparison opts itself out below 20 records
    # (metrics.MIN_CLIMATOLOGY_RECORDS).
    "global_1deg": dict(n_steps=300, record_interval=25),
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


def _now_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_failure_record(variant, run_id, exc, solver_atol=DEFAULT_SOLVER_ATOL):
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
        run_id=np.asarray(run_id),
        generated_at=np.asarray(_now_stamp()),
        run_config_json=np.asarray(json.dumps(dict(n_steps=cfg["n_steps"], record_interval=cfg["record_interval"]))),
        status=np.asarray("error"),
        error_message=np.asarray(f"{type(exc).__name__}: {exc}"),
        real_diverged_at=np.asarray(-1),
        mini_nonfinite_at=np.asarray(-1),
        steps_completed=np.asarray(0),
        solver_atol=np.asarray(solver_atol),
    )
    out_path = result_path(RESULTS_DIR, run_id, name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"    recorded failure in {out_path}")


def run_variant(variant, veros_path, run_id, store_all_fields=False, solver_atol=DEFAULT_SOLVER_ATOL,
                device="cpu", snapshot_surface_only=False):
    name, family, overrides = variant["name"], variant["family"], variant["overrides"]
    group = FAMILIES[family]["group"]
    cfg = variant.get("run_config", RUN_CONFIG[group])
    n_steps, record_interval = cfg["n_steps"], cfg["record_interval"]

    print(f"--- {name} ({group}): {n_steps} steps, recording every {record_interval}, "
          f"solver atol {solver_atol:g}, device {device} ---")

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
        run_id=np.asarray(run_id),
        generated_at=np.asarray(_now_stamp()),
        run_config_json=np.asarray(json.dumps(dict(n_steps=n_steps, record_interval=record_interval))),
        status=np.asarray(status),
        error_message=np.asarray(""),
        real_diverged_at=np.asarray(-1 if real_diverged_at is None else real_diverged_at),
        mini_nonfinite_at=np.asarray(-1 if mini_nonfinite is None else mini_nonfinite[0]),
        steps_completed=np.asarray(timesteps[-1] if timesteps else 0),
        solver_atol=np.asarray(solver_atol),
        device=np.asarray(device),
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

    def frames(states, field):
        stacked = np.stack([s[field] for s in states])
        # the report's gifs only ever render the uppermost level of a 3D
        # field, so on a big grid the rest is stored and never looked at:
        # 360x160x60 float64 is 27.6 MB per record per side, the surface
        # slice 0.46 MB. Metrics are unaffected -- they are reduced in-process
        # from the full states before this runs.
        if snapshot_surface_only and stacked.ndim == 4:
            return stacked[:, :, :, -1]
        return stacked

    for field in wanted:
        out[f"{field}_mini_frames"] = frames(mini_states, field)
        out[f"{field}_real_frames"] = frames(real_states, field)
    # so a reader can tell whether a metric is recomputable from this file
    out["stored_fields"] = np.asarray(wanted)
    out["snapshot_surface_only"] = np.asarray(bool(snapshot_surface_only))

    out_path = result_path(RESULTS_DIR, run_id, name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--snapshot-surface-only", action="store_true",
                        help="store only the uppermost level of 3D snapshot fields. The report's gifs "
                             "render nothing else, and metrics are reduced from the full states before "
                             "storage, so this costs nothing but shrinks the frames by nz -- 27.6 MB to "
                             "0.46 MB per record per side at 360x160x60.")
    parser.add_argument("--store-all-fields", action="store_true",
                        help="store every prognostic field's frames, not just temp/psi. Metrics are "
                             "reduced in-process, so by default a new or corrected metric can only be "
                             "recomputed offline for temp/psi and needs a full rerun for the rest; this "
                             "makes the .npz self-contained, at roughly 3.5x the size.")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                        help="jax platform for BOTH codes (default cpu). float64 is required by the "
                             "comparison and is throttled 1:32-1:64 on grenoble's workstation-class "
                             "GPUs, but at these grid sizes both codes look dispatch-bound rather "
                             "than FLOP-bound -- measure, do not assume.")
    parser.add_argument("--solver-atol", type=float, default=DEFAULT_SOLVER_ATOL,
                        help="absolute residual bound forced on BOTH codes' bicgstab for the external "
                             f"mode (default {DEFAULT_SOLVER_ATOL:g}, what both codes ship with). "
                             "Tightening it shrinks the largest avoidable seed of their divergence and "
                             "costs only 1-3%% in wall time, but do NOT go below ~1e-12: the achievable "
                             "residual floor for this preconditioned system is around 1e-12, and asking "
                             "for less makes the stopping rule unsatisfiable. jax's bicgstab then "
                             "iterates past a stagnated residual into breakdown, where rho or omega "
                             "underflow to zero and the divisions in _bicgstab_solve poison x with NaN "
                             "*before* its k=-10/-11 sentinel stops the loop -- and both codes discard "
                             "the returned info. That is what killed acc_biharmonic_mixing on GPU at "
                             "1e-14 while CPU survived.")
    parser.add_argument("--run-id", default=None,
                        help="identifier for this run, used as the results filename segment and as the "
                             "id plot_matrix_report.py renders. Under a wandb sweep this is the sweep id, "
                             "so every agent lands on one set without coordinating. Defaults to "
                             "'local-<UTC timestamp>'.")
    args = parser.parse_args()

    if not (args.veros_path / "veros" / "__init__.py").exists():
        parser.error(f"no veros package found at {args.veros_path}")

    configure_veros_runtime(args.veros_path, device=args.device)

    if args.variant:
        selected = [VARIANTS_BY_NAME[args.variant]]
    elif args.variants:
        selected = [VARIANTS_BY_NAME[v] for v in args.variants]
    elif args.group:
        selected = [v for v in VARIANTS if FAMILIES[v["family"]]["group"] == args.group]
    else:
        selected = VARIANTS

    # one id for the whole invocation, so a full run's variants land in one
    # addressable set; a partial rerun (--variant/--group) gets its own unless
    # --run-id pins it to an existing set
    run_id = args.run_id or f"local-{_now_stamp()}"
    try:
        validate_run_id(run_id)
    except ValueError as e:
        parser.error(str(e))

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
            run_variant(variant, args.veros_path, run_id, args.store_all_fields, args.solver_atol,
                        args.device, args.snapshot_surface_only)
        except Exception as e:
            # Don't let one variant's failure abort the rest of the matrix --
            # but do leave a file behind saying so. Printing only (the old
            # behaviour) wrote no .npz, and plot_matrix_report.py's "latest"
            # resolution then silently fell back to that variant's newest
            # older snapshot: a 4-step smoke run, which passed the tolerance
            # gate purely because 4 steps is not enough time to diverge.
            print(f"    FAILED: {variant['name']}: {type(e).__name__}: {e}")
            traceback.print_exc()
            write_failure_record(variant, run_id, e, args.solver_atol)


if __name__ == "__main__":
    main()
