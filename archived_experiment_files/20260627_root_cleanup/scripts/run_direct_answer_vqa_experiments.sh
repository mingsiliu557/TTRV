#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR="$ROOT_DIR/verl"
DATA_ROOT="${DATA_ROOT:-/jiigan-hp/ttrv-datasets}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/vepfs_default/chanxueyan/lhp/lms/ttrv_runtime}"
RUN_TAG="${RUN_TAG:-direct_answer_vqa_$(date +%Y%m%d_%H%M%S)}"
DATASETS=(${DATASETS:-visualsimpleqa ocrbench_v1 aokvqa_da_val})
RUN_KIND="${RUN_KIND:-method}" # method or step0

cd "$VERL_DIR"

export PATH="/vepfs_default/chanxueyan/lhp/lms/envs/ttrv/bin:$PATH"
export PYTHON_BIN="${PYTHON_BIN:-/vepfs_default/chanxueyan/lhp/lms/envs/ttrv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NO_GPU="${NO_GPU:-2}"
export DATA_LOCAL_DIR="${DATA_LOCAL_DIR:-$DATA_ROOT/verl_data}"
export BACKBONE_PATH="${BACKBONE_PATH:-$RUNTIME_ROOT/models/OpenGVLab/InternVL3-2B}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$RUNTIME_ROOT/hf_home/transformers}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$RUNTIME_ROOT/hf_home/modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_ROOT/xdg_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export USE_RUNTIME_BACKBONE_MIRROR="${USE_RUNTIME_BACKBONE_MIRROR:-1}"
export RUNTIME_BACKBONE_COPY_WEIGHTS="${RUNTIME_BACKBONE_COPY_WEIGHTS:-1}"

export EPISODE="${EPISODE:-2}"
export N=1
export MINI_BATCH_SIZE=1
export MICRO_BATCH_SIZE=2
export N_VOTES_PER_PROMPT=32
export N_SAMPLES_PER_PROMPT=16
export ANSWER_PARSE_MODE=direct_answer_short
export ANSWER_CHOICE_LABELS=A-D
export TTRL_REWARD_STYLE="${TTRL_REWARD_STYLE:-density_cluster_soft}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
export UNKNOWN_REWARD=0.0
export ALL_UNKNOWN_REWARD=0.0
export DENSITY_TEMPERATURE_T0="${DENSITY_TEMPERATURE_T0:-0.20}"
export DENSITY_TEMPERATURE_T_MIN="${DENSITY_TEMPERATURE_T_MIN:-0.10}"
export DENSITY_TEMPERATURE_T_MAX="${DENSITY_TEMPERATURE_T_MAX:-0.30}"
export DENSITY_EMBEDDING_SCOPE="${DENSITY_EMBEDDING_SCOPE:-evidence_query_mean_pool}"
export TTRL_LOG_RESPONSE_EMBEDDINGS=0
export USE_PAPO_PRCP_LOSS="${USE_PAPO_PRCP_LOSS:-true}"
export PAPO_VALID_ONLY="${PAPO_VALID_ONLY:-false}"
export PAPO_KL_PRCP_CLIP="${PAPO_KL_PRCP_CLIP:-0.2}"
export PAPO_MASK_PATCH_SIZE="${PAPO_MASK_PATCH_SIZE:-14}"
export PAPO_MASK_TYPE="${PAPO_MASK_TYPE:-black}"
export ACTOR_CLIP_RATIO_LOW="${ACTOR_CLIP_RATIO_LOW:-0.2}"
export ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}"
export ROLLOUT_LIMIT_IMAGES="${ROLLOUT_LIMIT_IMAGES:-1}"
export TTRL_VAL_NUM_EXAMINE="${TTRL_VAL_NUM_EXAMINE:-0}"

run_one() {
  local key="$1"
  local task tag mask_prob kl_coef ori_nll actor_kl clip_high
  case "$key" in
    visualsimpleqa)
      task="${TASK_OVERRIDE:-visualsimpleqa_da_20_hq_major_correct}"
      tag="visualsimpleqa_da"
      mask_prob="${PAPO_MASK_PROB:-0.6}"
      kl_coef="${PAPO_KL_PRCP_COEF:-0.01}"
      ori_nll="${PAPO_ORI_ENTROPY_COEF:-0.0}"
      actor_kl="${ACTOR_KL_LOSS_COEF:-0.01}"
      clip_high="${ACTOR_CLIP_RATIO_HIGH:-0.3}"
      ;;
    ocrbench_v1)
      task="${TASK_OVERRIDE:-ocrbench_v1_20_hq_major_correct}"
      tag="ocrbench_v1"
      mask_prob="${PAPO_MASK_PROB:-0.45}"
      kl_coef="${PAPO_KL_PRCP_COEF:-0.01}"
      ori_nll="${PAPO_ORI_ENTROPY_COEF:-0.0}"
      actor_kl="${ACTOR_KL_LOSS_COEF:-0.01}"
      clip_high="${ACTOR_CLIP_RATIO_HIGH:-0.3}"
      ;;
    aokvqa_da_val)
      task="${TASK_OVERRIDE:-aokvqa_da_val_20_hq_major_correct}"
      tag="aokvqa_da_val"
      mask_prob="${PAPO_MASK_PROB:-0.6}"
      kl_coef="${PAPO_KL_PRCP_COEF:-0.01}"
      ori_nll="${PAPO_ORI_ENTROPY_COEF:-0.0}"
      actor_kl="${ACTOR_KL_LOSS_COEF:-0.01}"
      clip_high="${ACTOR_CLIP_RATIO_HIGH:-0.3}"
      ;;
    *)
      echo "unknown dataset key: $key" >&2
      exit 2
      ;;
  esac

  export TASK="$task"
  export PAPO_MASK_PROB="$mask_prob"
  export PAPO_KL_PRCP_COEF="$kl_coef"
  export PAPO_ORI_ENTROPY_COEF="$ori_nll"
  export ACTOR_KL_LOSS_COEF="$actor_kl"
  export ACTOR_CLIP_RATIO_HIGH="$clip_high"

  local run_name="${tag}_${RUN_KIND}_${RUN_TAG}"
  local run_dir="$DATA_ROOT/experiments/$run_name"
  mkdir -p "$run_dir"
  export LOG_DIR="$run_dir"
  export LOG_FILE="$run_dir/run.log"
  export TTRL_EVAL_OUTPUT_JSONL="$run_dir/validation_flat.jsonl"
  export TTRL_TRAIN_ROLLOUT_JSONL="$run_dir/train_rollouts.jsonl"
  unset TTRL_EVAL_GROUP_OUTPUT_JSONL

  echo "[direct-answer-run] key=$key kind=$RUN_KIND task=$TASK run_dir=$run_dir"
  echo "[direct-answer-run] reward=$TTRL_REWARD_STYLE parser=$ANSWER_PARSE_MODE mask=$PAPO_MASK_PROB kl=$PAPO_KL_PRCP_COEF actor_kl=$ACTOR_KL_LOSS_COEF"

  if [[ "$RUN_KIND" == "step0" ]]; then
    bash examples/ttrv/run.sh \
      trainer.val_before_train=True \
      +trainer.val_only=True \
      trainer.test_freq=0 \
      trainer.total_epochs=0 \
      trainer.save_freq=0 \
      trainer.max_actor_ckpt_to_keep=0 \
      trainer.max_critic_ckpt_to_keep=0 \
      trainer.experiment_name="$run_name" \
      trainer.default_local_dir="$DATA_ROOT/checkpoints/disabled/$run_name"
  else
    bash examples/ttrv/run.sh \
      trainer.val_before_train=False \
      trainer.test_freq=10 \
      trainer.save_freq=0 \
      trainer.max_actor_ckpt_to_keep=0 \
      trainer.max_critic_ckpt_to_keep=0 \
      trainer.experiment_name="$run_name" \
      trainer.default_local_dir="$DATA_ROOT/checkpoints/disabled/$run_name"
  fi
}

for key in "${DATASETS[@]}"; do
  run_one "$key"
done

echo "[direct-answer-run] complete kind=$RUN_KIND tag=$RUN_TAG"
