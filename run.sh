#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TTRV"
VERL_DIR="$ROOT_DIR/verl"
cd "$VERL_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NO_GPU="${NO_GPU:-4}"
export TASK="${TASK:-dtd_20}"
export EPISODE="${EPISODE:-2}"
export DATA_LOCAL_DIR="${DATA_LOCAL_DIR:-$VERL_DIR/data}"
export BACKBONE_PATH="${BACKBONE_PATH:-OpenGVLab/InternVL3-2B}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export ANSWER_PARSE_MODE="${ANSWER_PARSE_MODE:-legacy}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TTRL_REWARD_STYLE=vision_self_harmony

if [ -x /root/miniconda3/envs/ttrv/bin/python ]; then
  export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/ttrv/bin/python}"
else
  export PYTHON_BIN="${PYTHON_BIN:-python}"
fi

RUN_TAG="${RUN_TAG:-$(date +%m%d_%H%M%S)}"
MASTER_LOG="$VERL_DIR/logs/dtd20_vsh_transform_ablation_${RUN_TAG}.log"
OUT_DIR="$VERL_DIR/outputs/dtd20_vsh_transform_ablation_${RUN_TAG}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"

mkdir -p "$VERL_DIR/logs" "$OUT_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

COMMON_ARGS=(
  trainer.val_before_train=False
  trainer.test_freq=200000
  trainer.save_freq=0
  trainer.max_actor_ckpt_to_keep=0
  trainer.max_critic_ckpt_to_keep=0
)

TRANSFORMS=(
  center_crop_s098
  center_crop_s095
  center_crop_s092
  center_crop_s088
  center_crop_s084
  photometric_weak
  photometric_medium
  photometric_strong
  cotta_weak_noflip
  cotta_strong_noflip
  multi_aug_safe
  cotta_strong_dtd_flip
)

echo "[master] Vision Self-Harmony transform ablation"
echo "[master] run_tag=$RUN_TAG"
echo "[master] log=$MASTER_LOG"
echo "[master] outputs=$OUT_DIR"
echo "[master] cuda=$CUDA_VISIBLE_DEVICES task=$TASK model=$BACKBONE_PATH"
echo "[master] votes=$N_VOTES_PER_PROMPT samples=$N_SAMPLES_PER_PROMPT parser=$ANSWER_PARSE_MODE"
echo "[master] auto_shutdown=$AUTO_SHUTDOWN"

run_one() {
  local transform="$1"
  local index="$2"
  local total="$3"
  echo "[$index/$total] vision_self_harmony transform=$transform"
  HARMONY_TRANSFORM_TYPE="$transform" \
  TTRL_TRAIN_ROLLOUT_JSONL="$OUT_DIR/${transform}_train_rollouts.jsonl" \
  TTRL_EVAL_OUTPUT_JSONL="$OUT_DIR/${transform}_final_eval_flat.jsonl" \
  bash examples/ttrv/run.sh "${COMMON_ARGS[@]}" \
    trainer.experiment_name="dtd20_vsh_${transform}_${RUN_TAG}" \
    trainer.default_local_dir="checkpoints/TTRL-verl/dtd20_vsh_transform_ablation_${RUN_TAG}/${transform}"
}

total="${#TRANSFORMS[@]}"
for i in "${!TRANSFORMS[@]}"; do
  run_one "${TRANSFORMS[$i]}" "$((i + 1))" "$total"
done

echo "[master] all transform ablations finished"
echo "[master] outputs=$OUT_DIR"

if [ "$AUTO_SHUTDOWN" = "1" ]; then
  echo "[master] shutting down now"
  shutdown -h now
else
  echo "[master] AUTO_SHUTDOWN=0, skip shutdown"
fi
