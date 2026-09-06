#!/usr/bin/env bash
# One chunk of the comparison matrix, as an OAR batch job on Grid5000.
#
# Every chunk is passed the SAME run id, so the whole matrix lands in one
# addressable set and `plot_matrix_report.py --run-id <id>` renders all of it.
# Unlike the wandb sweep path -- where the sweep id supplies that identifier
# for free -- here you have to pick one and pass it to every chunk yourself.
#
# Usage (from the repo root on the frontend):
#   oarsub -l nodes=1,walltime=8:00:00 \
#          "./test/sweep/run_matrix_oar.sh local-20260901T220000Z acc_basic acc_full ..."
#
# See guide_to_run.md for why this is a plain OAR job rather than `g5k launch`.

set -euo pipefail

REPO=/home/emeunier/code/MiniVeros-Autodiff
CONDA_ENV=veros

if [ $# -lt 2 ]; then
    echo "usage: $0 <run-id> <variant> [variant ...]" >&2
    exit 2
fi

RUN_ID="$1"
shift

# shellcheck disable=SC1090
source ~/.bash_profile
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

cd "$REPO"

: "${STORE:?STORE is not set -- every script here writes to \$STORE/MiniVeros-Autodiff}"
echo "host=$(hostname) store=$STORE run_id=$RUN_ID"
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
# DEVICE=gpu runs both codes on the GPU. The job must then be reserved with
# gpu=1, or jax falls back to CPU and the run is silently mislabelled.
if [ -n "${DEVICE:-}" ]; then
    EXTRA+=(--device "$DEVICE")
    echo "device=$DEVICE"
    if [ "$DEVICE" = "gpu" ]; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -2 || echo "no nvidia-smi"
        python -c "import jax; d=jax.devices(); print('jax devices:', d); assert d[0].platform=='gpu', 'jax fell back to CPU'" || exit 1
    fi
fi

# Horizon and storage overrides, for a family whose RUN_CONFIG default is not
# what this particular job wants (e.g. a year-long 1-degree run).
if [ -n "${STEPS:-}" ]; then
    EXTRA+=(--steps "$STEPS")
    echo "steps=$STEPS"
fi
if [ -n "${RECORD_INTERVAL:-}" ]; then
    EXTRA+=(--record-interval "$RECORD_INTERVAL")
    echo "record_interval=$RECORD_INTERVAL"
fi
if [ "${SNAPSHOT_SURFACE_ONLY:-0}" != "0" ]; then
    EXTRA+=(--snapshot-surface-only)
    echo "storing surface slices only"
fi

# -u: OAR redirects stdout to a file, so without it Python buffers and the
# log stays empty for the whole job -- there is no way to tell a slow chunk
# from a stuck one.
python -u test/generate_matrix_data.py --run-id "$RUN_ID" "${EXTRA[@]}" --variants "$@"

echo "chunk done: $*"
