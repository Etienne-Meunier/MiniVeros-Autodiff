#!/usr/bin/env bash
# One chunk of the comparison matrix, as an OAR batch job on Grid5000.
#
# Every chunk of a sweep is passed the SAME run timestamp, so the whole
# matrix lands on one snapshot and `plot_matrix_report.py --strict` has a
# single timestamp to render. That is the point: with per-job timestamps a
# variant that crashed in one chunk silently falls back to an older, shorter
# snapshot of itself and shows up in the report as a pass.
#
# Usage (from the repo root on the frontend):
#   oarsub -l nodes=1,walltime=8:00:00 \
#          "./test/sweep/run_matrix_oar.sh 20260901T220000Z acc_basic acc_full ..."
#
# See guide_to_run.md for why this is a plain OAR job rather than `g5k launch`.

set -euo pipefail

REPO=/home/emeunier/code/MiniVeros-Autodiff
CONDA_ENV=veros

if [ $# -lt 2 ]; then
    echo "usage: $0 <run-timestamp> <variant> [variant ...]" >&2
    exit 2
fi

RUN_TIMESTAMP="$1"
shift

# shellcheck disable=SC1090
source ~/.bash_profile
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

cd "$REPO"

: "${STORE:?STORE is not set -- every script here writes to \$STORE/MiniVeros-Autodiff}"
echo "host=$(hostname) store=$STORE timestamp=$RUN_TIMESTAMP"
echo "variants: $*"

# STORE_ALL_FIELDS=1 keeps every prognostic field's frames rather than just
# temp/psi, so a metric can be recomputed offline instead of costing a rerun.
# Passed by environment because the argument list here is positional.
EXTRA=()
if [ "${STORE_ALL_FIELDS:-0}" != "0" ]; then
    EXTRA+=(--store-all-fields)
    echo "storing all prognostic fields (~3.5x the .npz size)"
fi
# SOLVER_ATOL overrides the tightened default; pass 1e-8 to reproduce the
# tolerance both codes ship with.
if [ -n "${SOLVER_ATOL:-}" ]; then
    EXTRA+=(--solver-atol "$SOLVER_ATOL")
    echo "forcing elliptic-solver atol=$SOLVER_ATOL"
fi

# -u: OAR redirects stdout to a file, so without it Python buffers and the
# log stays empty for the whole job -- there is no way to tell a slow chunk
# from a stuck one.
python -u test/generate_matrix_data.py --run-timestamp "$RUN_TIMESTAMP" "${EXTRA[@]}" --variants "$@"

echo "chunk done: $*"
