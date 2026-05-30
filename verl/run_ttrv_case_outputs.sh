#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

RUN_TAG="${RUN_TAG:-$(date +%m%d_%H%M%S)}"
CASE_DIR="$SCRIPT_DIR/outputs/dtd20_case_ttrv_${RUN_TAG}"
mkdir -p "$CASE_DIR" "$SCRIPT_DIR/logs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NO_GPU="${NO_GPU:-4}"
export TASK="${TASK:-dtd_20}"
export EPISODE="${EPISODE:-2}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export N="${N:-32}"
export ANSWER_PARSE_MODE="${ANSWER_PARSE_MODE:-legacy}"
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/ttrv/bin/python}"

COMMON_CASE_ARGS=(
  "data.val_files=[$SCRIPT_DIR/data/dtd_20/train.parquet]"
  trainer.save_freq=0
  trainer.max_actor_ckpt_to_keep=0
  trainer.max_critic_ckpt_to_keep=0
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.n=32
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95
)

echo "[case] run_tag=$RUN_TAG"
echo "[case] outputs=$CASE_DIR"

echo "[case] base step0 sampled eval on dtd_20/train.parquet"
TTRL_REWARD_STYLE=frequency_entropy ENTROPY_COEF=0.75 \
TTRL_EVAL_OUTPUT_JSONL="$CASE_DIR/base_step0_eval_flat.jsonl" \
TTRL_EVAL_GROUP_OUTPUT_JSONL="$CASE_DIR/base_step0_eval_groups.jsonl" \
bash examples/ttrv/run.sh "${COMMON_CASE_ARGS[@]}" \
  trainer.val_before_train=True \
  +trainer.val_only=True \
  trainer.test_freq=0 \
  trainer.total_epochs=0 \
  trainer.experiment_name="case_base_step0_${RUN_TAG}" \
  trainer.default_local_dir="checkpoints/TTRL-verl/dtd20_case_${RUN_TAG}/base_step0"

echo "[case] TTRV official train + sampled final eval on dtd_20/train.parquet"
TTRL_REWARD_STYLE=frequency_entropy ENTROPY_COEF=0.75 \
TTRL_TRAIN_ROLLOUT_JSONL="$CASE_DIR/ttrv_train_rollouts.jsonl" \
TTRL_EVAL_OUTPUT_JSONL="$CASE_DIR/ttrv_final_eval_flat.jsonl" \
TTRL_EVAL_GROUP_OUTPUT_JSONL="$CASE_DIR/ttrv_final_eval_groups.jsonl" \
bash examples/ttrv/run.sh "${COMMON_CASE_ARGS[@]}" \
  trainer.val_before_train=False \
  trainer.test_freq=200000 \
  trainer.experiment_name="case_ttrv_official_${RUN_TAG}" \
  trainer.default_local_dir="checkpoints/TTRL-verl/dtd20_case_${RUN_TAG}/ttrv_official"

echo "[case] done"
