#!/usr/bin/env bash
# A long single-code integration as an OAR batch job. CODE=mini|veros.
#
# Companion to run_matrix_oar.sh, and written as a file in the repo for the
# same reason: it has to be synced by unison, not composed on the remote.
# A heredoc-over-ssh mangles the `eval "$(conda shell.bash hook)"` line into
# `eval "\$(...)"`, conda never activates, and the job dies with `python:
# command not found` after OAR has already burned the reservation.
#
# Usage (from the frontend):
#   oarsub -l gpu=1,walltime=24:00:00 -p "gpu_count > 0" -n mv10y \
#     -O ~/oar_logs/long10y.out -E ~/oar_logs/long10y.err \
#     "env YEARS=10 RUN_TIMESTAMP=<TS> bash ~/code/MiniVeros-Autodiff/test/sweep/run_long_mini_oar.sh"
#
# Invoke as `bash <path>`: unison does not carry the executable bit.

set -uo pipefail

REPO=/home/emeunier/code/MiniVeros-Autodiff
CONDA_ENV=veros

VARIANT=${VARIANT:-global_1deg}
CODE=${CODE:-mini}
YEARS=${YEARS:-10}
SEGMENT_YEARS=${SEGMENT_YEARS:-0.5}
LOG_DAYS=${LOG_DAYS:-30.4}
DEVICE=${DEVICE:-gpu}

# shellcheck disable=SC1090
source ~/.bash_profile
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
cd "$REPO"

: "${STORE:?STORE is not set -- run_long.py writes to \$STORE/MiniVeros-Autodiff}"
echo "host=$(hostname) store=$STORE"
if [ "$DEVICE" = "gpu" ]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1 || echo "no nvidia-smi"
    # fail loudly rather than silently producing a CPU run labelled gpu
    python -c "import jax; d=jax.devices(); print('jax devices:', d); assert d[0].platform=='gpu', 'jax fell back to CPU'" || exit 1
fi

EXTRA=()
if [ -n "${RUN_TIMESTAMP:-}" ]; then
    EXTRA+=(--run-timestamp "$RUN_TIMESTAMP")
fi
if [ -n "${SOLVER_ATOL:-}" ]; then
    EXTRA+=(--solver-atol "$SOLVER_ATOL")
fi

# -u: OAR redirects stdout to a file, so without it Python buffers and a
# 17-hour job looks identical to a stuck one.
python -u test/run_long.py \
    --code "$CODE" --variant "$VARIANT" --years "$YEARS" --segment-years "$SEGMENT_YEARS" \
    --log-days "$LOG_DAYS" --device "$DEVICE" "${EXTRA[@]}"

echo "long run done: $CODE, $VARIANT, $YEARS years"
