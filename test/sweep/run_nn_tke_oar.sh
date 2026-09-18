#!/usr/bin/env bash
# One or more test/nn_tke experiment scripts, in sequence, as an OAR batch job.
# Mirrors run_paper_oar.sh (same reasons: unison-synced file, not composed on the remote).
#
# exp10d and exp10e both load exp10c's serialised nets from $STORE/.../results/nn_tke/, so
# exp10c must run (and finish) before either -- pass them in dependency order, e.g.:
#
#   oarsub -l gpu=1,walltime=6:00:00 -p "gpu_count > 0" -n mv-nntke \
#     -O ~/oar_logs/nntke.out -E ~/oar_logs/nntke.err \
#     "bash ~/code/MiniVeros-Autodiff/test/sweep/run_nn_tke_oar.sh exp10b_gradient_check exp10c_offline_pretrain exp10d_online_finetune exp10e_calibration_limit"
#
# Invoke as `bash <path>`: unison does not carry the executable bit.

set -uo pipefail

REPO=/home/emeunier/code/MiniVeros-Autodiff
CONDA_ENV=veros
DEVICE=${DEVICE:-gpu}

if [ $# -lt 1 ]; then
    echo "usage: $0 <expN_name> [<expN_name> ...]" >&2
    exit 2
fi

# shellcheck disable=SC1090
source ~/.bash_profile
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
cd "$REPO"

: "${STORE:?STORE is not set -- common.py writes to \$STORE/MiniVeros-Autodiff/results/nn_tke}"
echo "host=$(hostname) store=$STORE device=$DEVICE"
if [ "$DEVICE" = "gpu" ]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1 || echo "no nvidia-smi"
    python -c "import jax; d=jax.devices(); print('jax devices:', d); assert d[0].platform=='gpu', 'jax fell back to CPU'" || exit 1
fi

cd test/nn_tke

for name in "$@"; do
    echo "--- $name"
    python -u "${name}.py"
done

echo "nn_tke run done: $*"
