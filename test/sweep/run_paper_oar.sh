#!/usr/bin/env bash
# One or more paper-figure experiment scripts, in sequence, as an OAR batch job.
#
# Companion to run_long_oar.sh and run_matrix_oar.sh, and written as a file in
# the repo for the same reason: it has to be synced by unison, not composed on
# the remote. A heredoc-over-ssh mangles the `eval "$(conda shell.bash hook)"`
# line into `eval "\$(...)"`, conda never activates, and the job dies with
# `python: command not found` after OAR has already burned the reservation.
#
# Used for the two steps that must run before anything else can: the spin-up
# (run this alone first -- agents started in parallel would all race to build
# the cache) and exp1_gradient_check (whose horizon H everything else needs).
# exp2 onward run better as the wandb sweep in paper.yaml, one script per agent.
#
# Usage (from the frontend):
#   oarsub -l gpu=1,walltime=8:00:00 -p "gpu_count > 0" -n mv-spinup \
#     -O ~/oar_logs/spinup.out -E ~/oar_logs/spinup.err \
#     "bash ~/code/MiniVeros-Autodiff/test/sweep/run_paper_oar.sh spinup_only"
#
#   oarsub -l gpu=1,walltime=4:00:00 -p "gpu_count > 0" -n mv-exp1 \
#     -O ~/oar_logs/exp1.out -E ~/oar_logs/exp1.err \
#     "bash ~/code/MiniVeros-Autodiff/test/sweep/run_paper_oar.sh exp1_gradient_check"
#
# Invoke as `bash <path>`: unison does not carry the executable bit.

set -uo pipefail

REPO=/home/emeunier/code/MiniVeros-Autodiff
CONDA_ENV=veros
DEVICE=${DEVICE:-gpu}

if [ $# -lt 1 ]; then
    echo "usage: $0 spinup_only|<expN_name> [<expN_name> ...]" >&2
    exit 2
fi

# shellcheck disable=SC1090
source ~/.bash_profile
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
cd "$REPO"

: "${STORE:?STORE is not set -- common.py writes to \$STORE/MiniVeros-Autodiff/results/paper}"
echo "host=$(hostname) store=$STORE device=$DEVICE"
if [ "$DEVICE" = "gpu" ]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1 || echo "no nvidia-smi"
    # fail loudly rather than silently producing a CPU run labelled gpu
    python -c "import jax; d=jax.devices(); print('jax devices:', d); assert d[0].platform=='gpu', 'jax fell back to CPU'" || exit 1
fi

# -u: OAR redirects stdout to a file, so without it Python buffers and a
# multi-hour job looks identical to a stuck one.
cd test/paper_figures

if [ "$1" = "spinup_only" ]; then
    python -u -c "import common; common.spinup()"
    echo "spin-up done"
    exit 0
fi

for name in "$@"; do
    echo "--- $name"
    python -u "${name}.py"
done

echo "paper run done: $*"
