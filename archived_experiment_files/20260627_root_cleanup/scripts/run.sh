#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

# Run this inside a 2-GPU volc interactive worker.
# No shutdown/poweroff command is used here.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NO_GPU="${NO_GPU:-2}"
export DATA_ROOT="${DATA_ROOT:-/jiigan-hp/ttrv-datasets}"
export DATA_LOCAL_DIR="${DATA_LOCAL_DIR:-$DATA_ROOT/verl_data}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_home}"
export BACKBONE_PATH="${BACKBONE_PATH:-$DATA_ROOT/models/OpenGVLab/InternVL3-2B}"
export PYTHON_BIN="${PYTHON_BIN:-/vepfs_default/chanxueyan/lhp/lms/envs/ttrv/bin/python}"

export TASK_NAME="${TASK_NAME:-mme_20}"
export EPISODE="${EPISODE:-2}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export ANSWER_PARSE_MODE="${ANSWER_PARSE_MODE:-legacy}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Experiment switchboard.
export RUN_OFFICIAL="${RUN_OFFICIAL:-1}"
export RUN_FREQUENCY_ONLY="${RUN_FREQUENCY_ONLY:-1}"
export RUN_VSH="${RUN_VSH:-1}"
export RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
export SAVE_CKPT="${SAVE_CKPT:-0}"
export VSH_TRANSFORMS="${VSH_TRANSFORMS:-center_crop_s095 center_crop_s088 photometric_weak multi_aug_safe cotta_strong_noflip}"

RUN_TAG="${RUN_TAG:-mme20_full_$(date +%Y%m%d_%H%M%S)}"
export RUN_TAG

echo "[run.sh] MME20 TTRV/Vision Self-Harmony run"
echo "[run.sh] run_tag=$RUN_TAG"
echo "[run.sh] data_root=$DATA_ROOT"
echo "[run.sh] backbone=$BACKBONE_PATH"
echo "[run.sh] cuda=$CUDA_VISIBLE_DEVICES no_gpu=$NO_GPU"
echo "[run.sh] official=$RUN_OFFICIAL frequency_only=$RUN_FREQUENCY_ONLY vsh=$RUN_VSH"
echo "[run.sh] vsh_transforms=$VSH_TRANSFORMS"
echo "[run.sh] save_ckpt=$SAVE_CKPT analysis=$RUN_ANALYSIS"

bash "$ROOT_DIR/run_mme20_ttrv_failure_analysis.sh"

echo "[run.sh] complete"
echo "[run.sh] outputs under: $DATA_ROOT/experiments/mme20_${RUN_TAG}"
echo "[run.sh] no shutdown command was run"
