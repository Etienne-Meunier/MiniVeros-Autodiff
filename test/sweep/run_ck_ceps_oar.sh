#!/usr/bin/env bash
# The c_k/c_eps ACC density sweep (test/ck_ceps_sweep/), as an OAR batch job.
#
# Written as a file in the repo (not composed inline over ssh) for the same
# reason as run_paper_oar.sh: a heredoc-over-ssh mangles the
# `eval "$(conda shell.bash hook)"` line and conda never activates.
#
# One job, sequential: 25 runs of 30 model-years each, ~a few minutes apiece
# on a GPU (forward-only, no autodiff) -- no wandb sweep/multi-agent needed.
#
# Usage (from the frontend, after `g5k sync code`):
#   oarsub -l gpu=1,walltime=6:00:00 -p "gpu_count > 0" -n mv-ck-ceps \
#     -O ~/oar_logs/ck_ceps.out -E ~/oar_logs/ck_ceps.err \
#     "bash ~/code/MiniVeros-Autodiff/test/sweep/run_ck_ceps_oar.sh"
#
# Invoke as `bash <path>`: unison does not carry the executable bit.

set -uo pipefail

REPO=/home/emeunier/code/MiniVeros-Autodiff
CONDA_ENV=veros

# shellcheck disable=SC1090
source ~/.bash_profile
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
cd "$REPO"

: "${STORE:?STORE is not set -- common.py writes to \$STORE/MiniVeros-Autodiff/results/ck_ceps_sweep}"
echo "host=$(hostname) store=$STORE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1 || echo "no nvidia-smi"
python -c "import jax; d=jax.devices(); print('jax devices:', d); assert d[0].platform=='gpu', 'jax fell back to CPU'" || exit 1

# -u: OAR redirects stdout to a file, so without it Python buffers and a
# multi-hour job looks identical to a stuck one.
cd test/ck_ceps_sweep
python -u run_sweep.py "$@"
echo "ck_ceps sweep done"
