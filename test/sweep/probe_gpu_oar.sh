#!/usr/bin/env bash
# Is a GPU worth it for this comparison? One short run per grid, per device.
#
# Not a sweep: the question is only ms/step, so the horizon is short and the
# results are thrown away. Both codes run float64 (the comparison rests on
# it), which grenoble's workstation-class GPUs throttle to 1:32-1:64 of
# float32 -- but at 23k-62k cells both look dispatch-bound rather than
# FLOP-bound, and the two codes differ structurally there: veros dispatches
# hundreds of ops per step from Python, mini_veros compiles the whole run
# into one lax.scan. So the two may move in opposite directions.
#
# Usage (from the frontend), on a node reserved with a GPU:
#   oarsub -l gpu=1,walltime=1:00:00 -p "gpu_count > 0" \
#          -O ~/oar_logs/gpu_probe.out -E ~/oar_logs/gpu_probe.err \
#          "bash ~/code/MiniVeros-Autodiff/test/sweep/probe_gpu_oar.sh"

set -uo pipefail

REPO=/home/emeunier/code/MiniVeros-Autodiff
CONDA_ENV=veros
STEPS_ACC=${STEPS_ACC:-60}
STEPS_GLOBAL=${STEPS_GLOBAL:-20}

# shellcheck disable=SC1090
source ~/.bash_profile
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
cd "$REPO"

echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || echo "no nvidia-smi"
python -u -c "import jax; print('jax devices:', jax.devices())" 2>&1 | tail -2

for DEVICE in cpu gpu; do
    for SPEC in "acc_basic $STEPS_ACC 30" "global_default $STEPS_GLOBAL 10"; do
        set -- $SPEC
        echo "=== $1 on $DEVICE ($2 steps) ==="
        # a throwaway run id per combination; these .npz are for timing only.
        # Hyphens, not underscores: '__' separates variant from id in the
        # filename, so validate_run_id rejects '_' -- including the ones in the
        # variant name being interpolated here.
        python -u test/generate_matrix_data.py \
            --variant "$1" --steps "$2" --record-interval "$3" \
            --device "$DEVICE" --run-id "gpuprobe-${DEVICE}-${1//_/-}" 2>&1 \
            | grep -E "^---|mini:|FAILED|Error|error" || true
    done
done

echo "probe done"
