#!/usr/bin/env bash

# keep the safety flags if available via inclusion
if [ -f ./train.env ]; then
    source ./train.env
fi
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG_DIR=${1:-"logs/HoloMotionLabDebug/rsl_rl_$(date +%Y%m%d_%H%M%S)"}

PYTHON_BIN=${PYTHON_BIN:-python}

${PYTHON_BIN} -u -m holomotion.src.training.train_motion_tracking_isaaclab_rsl_rl \
    --log_dir "${LOG_DIR}" \
    --headless
