#!/usr/bin/env bash
# The top-5 (c_k, c_eps) full-state reruns (test/ck_ceps_sweep/run_top5_full_state.py),
# as an OAR batch job. Same reasoning as run_ck_ceps_oar.sh (file in the repo, not
# composed inline over ssh -- a heredoc mangles the conda hook line).
#
# 5 runs of 30 model-years, ~2GB of full-state .npz output total -- one job, no
# wandb sweep/multi-agent needed.
#
# Usage (from the frontend, after `g5k sync code`):
#   oarsub -l gpu=1,walltime=2:00:00 -p "gpu_count > 0" -n mv-top5-full \
#     -O ~/oar_logs/top5_full.out -E ~/oar_logs/top5_full.err \
#     "bash ~/code/MiniVeros-Autodiff/test/sweep/run_top5_full_state_oar.sh"
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

cd test/ck_ceps_sweep
python -u run_top5_full_state.py "$@"
echo "top5 full-state run done"
