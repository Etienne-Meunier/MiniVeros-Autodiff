#!/usr/bin/env python3
"""
The results layout, in one place: "results/{run_id}/{variant}.npz".

A run is a directory. Every report renders exactly one run id and nothing
else -- there is deliberately no "newest file wins" resolution anywhere, since
that rule is what let a variant missing from the current run get backfilled
from an older, shorter one, and a 4-step smoke run passes the tolerance gate
only because 4 steps is not enough time to diverge.

One directory per run also matches what `g5k sync model <run_id>` expects, so
pulling a single run off the cluster is one command instead of a hand-written
rsync glob. The older layout was flat -- "results/{variant}__{run_id}.npz" --
see migrate_results_layout.py.

Kept free of veros/jax imports so the generator and all three plotters can
share it.
"""

import re

# A run id is a directory name. It must not escape results/, and must not
# collide with the sibling directories that are not runs.
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Subdirectories of results/ that are not runs.
NOT_RUNS = {"divergence"}


def validate_run_id(run_id):
    if not RUN_ID_RE.match(run_id) or run_id in NOT_RUNS:
        raise ValueError(
            f"invalid run id {run_id!r}: it is a directory name under results/, so use letters, "
            f"digits, '.', '_' and '-', starting with a letter or digit, and not one of "
            f"{sorted(NOT_RUNS)}. A wandb sweep id or a label like 'local-20260906T120000Z' "
            f"both qualify."
        )
    return run_id


def run_dir(results_dir, run_id):
    return results_dir / run_id


def result_path(results_dir, run_id, variant):
    return results_dir / run_id / f"{variant}.npz"


def legacy_flat_files(results_dir):
    """Results still in the old "{variant}__{run_id}.npz" flat layout.

    run_long.py keeps its own flat "longrun_*" family and is not part of this.
    """
    return sorted(p for p in results_dir.glob("*__*.npz") if not p.name.startswith("longrun_"))


def available_run_ids(results_dir):
    """[(run_id, {"variants": n, "generated_at": stamp or None})], newest first.

    `generated_at` is provenance read out of the files -- it orders this
    listing for human convenience and nothing else.
    """
    import numpy as np

    out = []
    for d in results_dir.iterdir() if results_dir.is_dir() else []:
        if not d.is_dir() or d.name in NOT_RUNS:
            continue
        files = sorted(d.glob("*.npz"))
        if not files:
            continue
        newest = None
        for path in files:
            try:
                with np.load(path) as data:
                    stamp = str(data["generated_at"]) if "generated_at" in data.files else None
            except Exception:
                stamp = None
            if stamp and (newest is None or stamp > newest):
                newest = stamp
        out.append((d.name, {"variants": len(files), "generated_at": newest}))
    return sorted(out, key=lambda kv: (kv[1]["generated_at"] or "", kv[0]), reverse=True)


def require_run_id(parser, results_dir, run_id, needs=None):
    """Fail with the ids that do exist, so a forgotten id is self-correcting
    rather than a dead end. `needs` names a variant the run must contain."""
    known = available_run_ids(results_dir)
    if run_id not in dict(known):
        listing = "\n".join(
            f"  {rid}  ({info['variants']} variants"
            + (f", generated {info['generated_at']}" if info["generated_at"] else "") + ")"
            for rid, info in known[:20]
        )
        stranded = legacy_flat_files(results_dir)
        hint = ""
        if stranded:
            hint = (
                f"\n{len(stranded)} result(s) are still in the old flat layout and are not "
                f"visible as runs. Move them with:\n"
                f"  python test/migrate_results_layout.py"
            )
        parser.error(
            f"no results for run id {run_id!r} in {results_dir}."
            + (f" Available run ids (newest first):\n{listing}" if known else " No runs at all.")
            + hint
        )
    if needs and not result_path(results_dir, run_id, needs).exists():
        parser.error(
            f"run {run_id!r} has no result for {needs!r} ({run_id}/{needs}.npz). "
            f"Run it into this id with `generate_matrix_data.py --variant {needs} --run-id {run_id}`."
        )
    return run_id
