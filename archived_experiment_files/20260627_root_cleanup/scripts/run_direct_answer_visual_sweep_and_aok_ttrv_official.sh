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
RUN_TAG="${RUN_TAG:-direct_answer_visual_sweep_aok_ttrv_$(date +%Y%m%d_%H%M%S)}"

launch_worker() {
  local name="$1"
  local inner_cmd="$2"
  echo "[submit] $name"
  volc ml_devinstance launch \
    --resource_queue_id "$RESOURCE_QUEUE_ID" \
    --flavor_id "$FLAVOR_ID" \
    zsh -lc "$inner_cmd"
}

run_direct_answer_worker() {
  local name="$1"
  local dataset_key="$2"
  local reward_style="$3"
  local use_papo="$4"
  local mask_prob="$5"
  local kl_coef="$6"
  local ori_entropy_coef="$7"
  local density_t0="$8"
  local density_tmin="$9"
  local density_tmax="${10}"
  local density_scope="${11}"

  launch_worker "$name" "
    set -Eeuo pipefail
    cd '$ROOT_DIR'
    export DATA_ROOT='$DATA_ROOT'
    export RUNTIME_ROOT='$RUNTIME_ROOT'
    export BACKBONE_PATH='$BACKBONE_PATH'
    export PYTHON_BIN='$PYTHON_BIN'
    export CUDA_VISIBLE_DEVICES=0,1
    export NO_GPU=2
    export RUN_KIND=method
    export RUN_TAG='${RUN_TAG}_${name}'
    export DATASETS='$dataset_key'
    export TTRL_REWARD_STYLE='$reward_style'
    export ENTROPY_COEF=0.75
    export USE_PAPO_PRCP_LOSS='$use_papo'
    export DENSITY_TEMPERATURE_T0='$density_t0'
    export DENSITY_TEMPERATURE_T_MIN='$density_tmin'
    export DENSITY_TEMPERATURE_T_MAX='$density_tmax'
    export DENSITY_EMBEDDING_SCOPE='$density_scope'
    export PAPO_MASK_PROB='$mask_prob'
    export PAPO_KL_PRCP_COEF='$kl_coef'
    export PAPO_ORI_ENTROPY_COEF='$ori_entropy_coef'
    export ACTOR_KL_LOSS_COEF=0.01
    export ACTOR_CLIP_RATIO_HIGH=0.3
    bash run_direct_answer_vqa_experiments.sh
  "
}

echo "[main] tag=$RUN_TAG"
echo "[main] stage 1: A-OKVQA TTRV official frequency_entropy"
run_direct_answer_worker \
  "aokvqa_ttrv_official_frequency_entropy" \
  "aokvqa_da_val" \
  "frequency_entropy" \
  "false" \
  "0.60" "0.01" "0.0" \
  "0.20" "0.10" "0.30" "response_mean_pool"

echo "[main] stage 2: VisualSimpleQA evidence-density PAPO sweep"
# Lower mask: less aggressive counterfactual pressure for open-ended visual QA.
run_direct_answer_worker \
  "visual_d2_mask030_kl001_ori000" \
  "visualsimpleqa" \
  "density_cluster_answer_entropy" \
  "true" \
  "0.30" "0.01" "0.0" \
  "0.20" "0.10" "0.30" "evidence_query_mean_pool"

# Middle mask: tests whether the default 0.60 masking is too destructive for VisualSimpleQA.
run_direct_answer_worker \
  "visual_d2_mask045_kl001_ori000" \
  "visualsimpleqa" \
  "density_cluster_answer_entropy" \
  "true" \
  "0.45" "0.01" "0.0" \
  "0.20" "0.10" "0.30" "evidence_query_mean_pool"

# Lower PAPO KL: keeps density reward but reduces counterfactual pressure.
run_direct_answer_worker \
  "visual_d2_mask060_kl0005_ori000" \
  "visualsimpleqa" \
  "density_cluster_answer_entropy" \
  "true" \
  "0.60" "0.005" "0.0" \
  "0.20" "0.10" "0.30" "evidence_query_mean_pool"

# Add original-output entropy/NLL stabilizer to see whether it reduces invalid/free-form drift.
run_direct_answer_worker \
  "visual_d2_mask060_kl001_ori002" \
  "visualsimpleqa" \
  "density_cluster_answer_entropy" \
  "true" \
  "0.60" "0.01" "0.02" \
  "0.20" "0.10" "0.30" "evidence_query_mean_pool"

# Fixed density temperature: checks whether answer-entropy temperature is hurting this open-ended task.
run_direct_answer_worker \
  "visual_d1_fixed_mask030_kl001_ori000" \
  "visualsimpleqa" \
  "density_cluster_soft" \
  "true" \
  "0.30" "0.01" "0.0" \
  "0.20" "0.20" "0.20" "evidence_query_mean_pool"

# No PAPO diagnostic: separates density reward failure from PAPO masking failure.
run_direct_answer_worker \
  "visual_d2_papo_off" \
  "visualsimpleqa" \
  "density_cluster_answer_entropy" \
  "false" \
  "0.60" "0.01" "0.0" \
  "0.20" "0.10" "0.30" "evidence_query_mean_pool"


echo "[main] complete tag=$RUN_TAG"
