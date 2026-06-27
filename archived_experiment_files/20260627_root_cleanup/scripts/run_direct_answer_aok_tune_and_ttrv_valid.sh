#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

RESOURCE_QUEUE_ID="${RESOURCE_QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"
DATA_ROOT="${DATA_ROOT:-/jiigan-hp/ttrv-datasets}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/vepfs_default/chanxueyan/lhp/lms/ttrv_runtime}"
BACKBONE_PATH="${BACKBONE_PATH:-$RUNTIME_ROOT/models/OpenGVLab/InternVL3-2B}"
PYTHON_BIN="${PYTHON_BIN:-/vepfs_default/chanxueyan/lhp/lms/envs/ttrv/bin/python}"
RUN_TAG="${RUN_TAG:-direct_answer_aok_tune_ttrv_valid_$(date +%Y%m%d_%H%M%S)}"

launch_worker() {
  local name="$1"
  local inner_cmd="$2"
  echo "[submit] $name"
  volc ml_devinstance launch \
    --resource_queue_id "$RESOURCE_QUEUE_ID" \
    --flavor_id "$FLAVOR_ID" \
    zsh -lc "$inner_cmd"
}

run_aok_tune() {
  local variant="$1"
  local mask_prob="$2"
  local kl_coef="$3"
  local ori_entropy_coef="$4"
  local clip_high="${5:-0.3}"

  launch_worker "aokvqa_tune_${variant}" "
    set -Eeuo pipefail
    cd '$ROOT_DIR'
    export DATA_ROOT='$DATA_ROOT'
    export RUNTIME_ROOT='$RUNTIME_ROOT'
    export BACKBONE_PATH='$BACKBONE_PATH'
    export PYTHON_BIN='$PYTHON_BIN'
    export CUDA_VISIBLE_DEVICES=0,1
    export NO_GPU=2
    export RUN_KIND=method
    export RUN_TAG='${RUN_TAG}_aokvqa_${variant}'
    export DATASETS='aokvqa_da_val'
    export TTRL_REWARD_STYLE=density_cluster_answer_entropy
    export USE_PAPO_PRCP_LOSS=true
    export DENSITY_TEMPERATURE_T0=0.20
    export DENSITY_TEMPERATURE_T_MIN=0.10
    export DENSITY_TEMPERATURE_T_MAX=0.30
    export DENSITY_EMBEDDING_SCOPE=evidence_query_mean_pool
    export PAPO_MASK_PROB='$mask_prob'
    export PAPO_KL_PRCP_COEF='$kl_coef'
    export PAPO_ORI_ENTROPY_COEF='$ori_entropy_coef'
    export ACTOR_KL_LOSS_COEF=0.01
    export ACTOR_CLIP_RATIO_HIGH='$clip_high'
    bash run_direct_answer_vqa_experiments.sh
  "
}

run_ttrv_valid_only() {
  local key="$1"
  launch_worker "ttrv_valid_only_${key}" "
    set -Eeuo pipefail
    cd '$ROOT_DIR'
    export DATA_ROOT='$DATA_ROOT'
    export RUNTIME_ROOT='$RUNTIME_ROOT'
    export BACKBONE_PATH='$BACKBONE_PATH'
    export PYTHON_BIN='$PYTHON_BIN'
    export CUDA_VISIBLE_DEVICES=0,1
    export NO_GPU=2
    export RUN_KIND=method
    export RUN_TAG='${RUN_TAG}_${key}_frequency_valid_entropy'
    export DATASETS='$key'
    export TTRL_REWARD_STYLE=frequency_valid_entropy
    export ENTROPY_COEF=0.75
    export USE_PAPO_PRCP_LOSS=false
    export DENSITY_EMBEDDING_SCOPE=response_mean_pool
    bash run_direct_answer_vqa_experiments.sh
  "
}

echo "[main] tag=$RUN_TAG"
echo "[main] stage 1: A-OKVQA density+PAPO tuning, fixed density temperature"
run_aok_tune "mask045_kl001_ori000" 0.45 0.01 0.0
run_aok_tune "mask030_kl001_ori000" 0.30 0.01 0.0
run_aok_tune "mask060_kl0005_ori000" 0.60 0.005 0.0
run_aok_tune "mask060_kl001_ori002" 0.60 0.01 0.02

echo "[main] stage 2: TTRV valid-only direct-answer baselines"
run_ttrv_valid_only "visualsimpleqa"
run_ttrv_valid_only "ocrbench_v1"

echo "[main] complete tag=$RUN_TAG"
