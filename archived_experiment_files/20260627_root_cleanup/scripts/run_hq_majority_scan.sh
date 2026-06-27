#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

TASK="${TASK:?TASK is required, e.g. crpe_20_hq_pool80_seed42}"
SOURCE_TASK="${SOURCE_TASK:?SOURCE_TASK is required, e.g. crpe_20}"
FINAL_TASK="${FINAL_TASK:?FINAL_TASK is required}"
ANSWER_CHOICE_LABELS="${ANSWER_CHOICE_LABELS:-A-D}"
ANSWER_PARSE_MODE="${ANSWER_PARSE_MODE:-legacy}"
HQ_METRIC="${HQ_METRIC:-exact}"
HQ_SMAPE_THRESHOLD="${HQ_SMAPE_THRESHOLD:-0.9}"
HQ_SELECTION_MODE="${HQ_SELECTION_MODE:-majority}"
HQ_REPEAT_FILL="${HQ_REPEAT_FILL:-0}"
RUN_TAG="${RUN_TAG:-hq_majority_scan_${TASK}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/jiigan-hp/ttrv-datasets/experiments/${RUN_TAG}}"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NO_GPU="${NO_GPU:-1}"
export DATA_LOCAL_DIR="${DATA_LOCAL_DIR:-/jiigan-hp/ttrv-datasets/verl_data}"
export HF_HOME="${HF_HOME:-/jiigan-hp/ttrv-datasets/hf_home}"
export BACKBONE_PATH="${BACKBONE_PATH:-/jiigan-hp/ttrv-datasets/models/OpenGVLab/InternVL3-2B}"
export PYTHON_BIN="${PYTHON_BIN:-/vepfs_default/chanxueyan/lhp/lms/envs/ttrv/bin/python}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export EPISODE=1
export MINI_BATCH_SIZE=1
export MICRO_BATCH_SIZE=1
export N_VOTES_PER_PROMPT=32
export N_SAMPLES_PER_PROMPT=16
export N=32
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
export TTRL_VAL_NUM_EXAMINE="${TTRL_VAL_NUM_EXAMINE:-0}"

GROUP_JSONL="$OUT_DIR/${TASK}_eval_groups.jsonl"
FLAT_JSONL="$OUT_DIR/${TASK}_eval_flat.jsonl"
LOG_FILE="$OUT_DIR/${TASK}.log"

echo "[hq-scan] task=$TASK source=$SOURCE_TASK final=$FINAL_TASK out=$OUT_DIR parser=$ANSWER_PARSE_MODE labels=$ANSWER_CHOICE_LABELS metric=$HQ_METRIC selection=$HQ_SELECTION_MODE repeat_fill=$HQ_REPEAT_FILL"
REPEAT_ARGS=()
if [[ "$HQ_REPEAT_FILL" == "1" || "$HQ_REPEAT_FILL" == "true" || "$HQ_REPEAT_FILL" == "True" ]]; then
  REPEAT_ARGS+=(--allow-repeat-fill)
fi
TTRL_REWARD_STYLE=frequency_entropy ENTROPY_COEF=0.75 \
TTRL_EVAL_OUTPUT_JSONL="$FLAT_JSONL" \
TTRL_EVAL_GROUP_OUTPUT_JSONL="$GROUP_JSONL" \
TASK="$TASK" ANSWER_PARSE_MODE="$ANSWER_PARSE_MODE" ANSWER_CHOICE_LABELS="$ANSWER_CHOICE_LABELS" LOG_FILE="$LOG_FILE" \
bash verl/examples/ttrv/run.sh \
  "data.val_files=[$DATA_LOCAL_DIR/$TASK/train.parquet]" \
  trainer.val_before_train=True \
  +trainer.val_only=True \
  trainer.test_freq=0 \
  trainer.total_epochs=0 \
  trainer.save_freq=0 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=32 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  trainer.experiment_name="${TASK}_${RUN_TAG}" \
  trainer.default_local_dir="/jiigan-hp/ttrv-datasets/checkpoints/hq_scan/${RUN_TAG}/${TASK}"

"$PYTHON_BIN" verl/scripts/prepare_hq_major_correct_split.py make-final \
  --pool-dir "$DATA_LOCAL_DIR/$TASK" \
  --scan-groups "$GROUP_JSONL" \
  --test-source-dir "$DATA_LOCAL_DIR/$SOURCE_TASK" \
  --output-dir "$DATA_LOCAL_DIR/$FINAL_TASK" \
  --final-size 20 \
  --metric "$HQ_METRIC" \
  --smape-threshold "$HQ_SMAPE_THRESHOLD" \
  --selection-mode "$HQ_SELECTION_MODE" \
  "${REPEAT_ARGS[@]}" | tee "$OUT_DIR/${FINAL_TASK}_prepare_summary.stdout"

echo "[hq-scan] complete final=$DATA_LOCAL_DIR/$FINAL_TASK out=$OUT_DIR"
