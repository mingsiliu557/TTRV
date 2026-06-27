#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-capture_crpe_val_smoke_$(date +%Y%m%d_%H%M%S)}"
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
export N_VOTES_PER_PROMPT=2
export N_SAMPLES_PER_PROMPT=1
export N=1
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
export TTRL_VAL_NUM_EXAMINE="${TTRL_VAL_NUM_EXAMINE:-1}"

COMMON_ARGS=(
  trainer.val_before_train=True
  +trainer.val_only=True
  trainer.test_freq=0
  trainer.total_epochs=0
  trainer.save_freq=0
  trainer.max_actor_ckpt_to_keep=0
  trainer.max_critic_ckpt_to_keep=0
  actor_rollout_ref.rollout.val_kwargs.do_sample=False
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55
)

run_one() {
  local task="$1"
  local labels="$2"
  local log_file="$OUT_DIR/${task}.log"
  echo "[smoke] task=$task labels=$labels log=$log_file out=$OUT_DIR"
  TASK="$task" ANSWER_PARSE_MODE=legacy ANSWER_CHOICE_LABELS="$labels"   TTRL_REWARD_STYLE=frequency_entropy ENTROPY_COEF=0.75   TTRL_EVAL_OUTPUT_JSONL="$OUT_DIR/${task}_eval_flat.jsonl"   TTRL_EVAL_GROUP_OUTPUT_JSONL="$OUT_DIR/${task}_eval_groups.jsonl"   LOG_FILE="$log_file"   bash verl/examples/ttrv/run.sh "${COMMON_ARGS[@]}"     trainer.experiment_name="${task}_${RUN_TAG}"     trainer.default_local_dir="/jiigan-hp/ttrv-datasets/checkpoints/smoke/${RUN_TAG}/${task}"
}

for smoke_task in ${SMOKE_TASKS:-capture crpe}; do
  case "$smoke_task" in
    capture) run_one capture_20_val_smoke2 A-D ;;
    crpe) run_one crpe_20_val_smoke2 A-D ;;
    *) echo "[smoke] unknown task: $smoke_task" >&2; exit 2 ;;
  esac
done

echo "[smoke] complete out=$OUT_DIR"
