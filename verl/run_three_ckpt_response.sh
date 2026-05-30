#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/autodl-tmp/TTRV/verl

RUN_TAG="${RUN_TAG:-$(date +%m%d_%H%M%S)}"

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
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

OUTPUT_DIR="outputs/dtd20_response_${RUN_TAG}"
CKPT_DIR="checkpoints/TTRL-verl/dtd20_ckpt_response_${RUN_TAG}"
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*.jsonl

COMMON_ARGS=(
  trainer.val_before_train=False
  trainer.test_freq=200000
  trainer.save_freq=20000000
  trainer.max_actor_ckpt_to_keep=1
  trainer.max_critic_ckpt_to_keep=0
)

echo "[run] RUN_TAG=$RUN_TAG"
echo "[run] output_dir=$OUTPUT_DIR"
echo "[run] ckpt_dir=$CKPT_DIR"

echo "[1/3] ttrv_official ckpt + final validation responses"
TTRL_EVAL_OUTPUT_JSONL="$OUTPUT_DIR/ttrv_official_step10_validation.jsonl" \
TTRL_REWARD_STYLE=frequency_entropy ENTROPY_COEF=0.75 \
bash examples/ttrv/run.sh "${COMMON_ARGS[@]}" \
  trainer.experiment_name="ckpt_response_ttrv_official_${RUN_TAG}" \
  trainer.default_local_dir="$CKPT_DIR/ttrv_official"

echo "[2/3] ttrv_no_entropy ckpt + final validation responses"
TTRL_EVAL_OUTPUT_JSONL="$OUTPUT_DIR/ttrv_no_entropy_step10_validation.jsonl" \
TTRL_REWARD_STYLE=frequency_entropy ENTROPY_COEF=0.0 \
bash examples/ttrv/run.sh "${COMMON_ARGS[@]}" \
  trainer.experiment_name="ckpt_response_ttrv_no_entropy_${RUN_TAG}" \
  trainer.default_local_dir="$CKPT_DIR/ttrv_no_entropy"

echo "[3/3] soft_pseudo_label ckpt + final validation responses"
TTRL_EVAL_OUTPUT_JSONL="$OUTPUT_DIR/soft_pseudo_label_step10_validation.jsonl" \
TTRL_REWARD_STYLE=soft_pseudo_label SOFT_LABEL_GAMMA=2.0 UNKNOWN_REWARD=0.0 ALL_UNKNOWN_REWARD=0.0 \
bash examples/ttrv/run.sh "${COMMON_ARGS[@]}" \
  trainer.experiment_name="ckpt_response_soft_pseudo_label_${RUN_TAG}" \
  trainer.default_local_dir="$CKPT_DIR/soft_pseudo_label"

echo "[done] RUN_TAG=$RUN_TAG"
