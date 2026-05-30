#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

RUN_TAG="${RUN_TAG:-$(date +%m%d_%H%M%S)}"
OUT_DIR="$SCRIPT_DIR/outputs/dtd20_vision_self_harmony_${RUN_TAG}"
mkdir -p "$OUT_DIR" "$SCRIPT_DIR/logs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NO_GPU="${NO_GPU:-4}"
export TASK="${TASK:-dtd_20}"
export EPISODE="${EPISODE:-2}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export ANSWER_PARSE_MODE="${ANSWER_PARSE_MODE:-legacy}"
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/ttrv/bin/python}"
export TTRL_REWARD_STYLE=vision_self_harmony

COMMON_ARGS=(
  trainer.val_before_train=False
  trainer.test_freq=200000
  trainer.save_freq=0
  trainer.max_actor_ckpt_to_keep=0
  trainer.max_critic_ckpt_to_keep=0
)

run_one() {
  local transform="$1"
  echo "[vision-self-harmony] transform=${transform} run_tag=${RUN_TAG}"
  HARMONY_TRANSFORM_TYPE="$transform" \
  TTRL_TRAIN_ROLLOUT_JSONL="$OUT_DIR/${transform}_train_rollouts.jsonl" \
  TTRL_EVAL_OUTPUT_JSONL="$OUT_DIR/${transform}_final_eval_flat.jsonl" \
  bash examples/ttrv/run.sh "${COMMON_ARGS[@]}" \
    trainer.experiment_name="vision_self_harmony_${transform}_${RUN_TAG}" \
    trainer.default_local_dir="checkpoints/TTRL-verl/dtd20_vision_self_harmony_${RUN_TAG}/${transform}"
}

run_one photometric
run_one center_crop_resize

echo "[vision-self-harmony] done run_tag=${RUN_TAG} outputs=${OUT_DIR}"
