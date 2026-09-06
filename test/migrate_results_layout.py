#!/usr/bin/env python3
"""
Move results from the old flat layout into one directory per run.

    results/{variant}__{run_id}.npz   ->   results/{run_id}/{variant}.npz

Run it once locally and once on the cluster (both hold a $STORE). It is
idempotent and it never overwrites: a file already present at the destination
is reported and left alone, so a half-finished migration can be re-run.

run_long.py's "longrun_*" files are a separate family with their own naming
and are not touched.

    python test/migrate_results_layout.py --dry-run    # show what would move
    python test/migrate_results_layout.py
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from run_ids import legacy_flat_files, validate_run_id

STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
RESULTS_DIR = STORE / "MiniVeros-Autodiff" / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the moves without making them")
    args = parser.parse_args()

    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"no results directory at {RESULTS_DIR}")

    stranded = legacy_flat_files(RESULTS_DIR)
    if not stranded:
        print(f"nothing to migrate in {RESULTS_DIR}")
        return

    moved = skipped = 0
    for path in stranded:
        variant, run_id = path.name[:-len(".npz")].split("__", 1)
        try:
            validate_run_id(run_id)
        except ValueError as e:
            # Old ids were only ever timestamps, so this should not fire; if it
            # does, say which file rather than inventing a name for it.
            print(f"SKIP {path.name}: {e}")
            skipped += 1
            continue

        dest = RESULTS_DIR / run_id / f"{variant}.npz"
        if dest.exists():
            print(f"SKIP {path.name}: {dest.relative_to(RESULTS_DIR)} already exists")
            skipped += 1
            continue

        print(f"{path.name} -> {dest.relative_to(RESULTS_DIR)}")
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            path.rename(dest)
        moved += 1

    verb = "would move" if args.dry_run else "moved"
    print(f"\n{verb} {moved} file(s), skipped {skipped}")


if __name__ == "__main__":
    main()
