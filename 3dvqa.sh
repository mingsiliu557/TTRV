#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TTRV"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)}"
NOHUP_LOG="${NOHUP_LOG:-$ROOT_DIR/3dvqa_${RUN_ID}.nohup.log}"
NOHUP_DISABLE="${NOHUP_DISABLE:-0}"

if [ "${TTRV_3DVQA_INNER:-0}" != "1" ] && [ "$NOHUP_DISABLE" != "1" ]; then
  echo "[launcher] starting detached 3D TTRV run"
  echo "[launcher] log: $NOHUP_LOG"
  TTRV_3DVQA_INNER=1 nohup bash "$0" "$@" >"$NOHUP_LOG" 2>&1 &
  echo "[launcher] pid: $!"
  exit 0
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/.cache}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/.cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/.cache/triton}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/root/autodl-tmp/.cache/wandb}"

NO_GPU="${NO_GPU:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-20}"
VAL_SUBSET="${VAL_SUBSET:-200}"
ROLLOUT_N="${ROLLOUT_N:-32}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
TEST_FREQ="${TEST_FREQ:-1}"
CASES="${CASES:-20}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
USE_KL_LOSS="${USE_KL_LOSS:-True}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.03}"
REWARD_ALPHA="${REWARD_ALPHA:-0.75}"
UNKNOWN_REWARD="${UNKNOWN_REWARD:--1.0}"
SHUTDOWN_ON_EXIT="${SHUTDOWN_ON_EXIT:-0}"
SHUTDOWN_DELAY_MINUTES="${SHUTDOWN_DELAY_MINUTES:-5}"
RUN_SUCCEEDED=0

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  echo
  echo "[cleanup] exit_code=$exit_code"
  if command -v ray >/dev/null 2>&1; then
    echo "[cleanup] stopping Ray..."
    ray stop --force >/dev/null 2>&1 || true
  fi

  if [ "$exit_code" -ne 0 ] || [ "$RUN_SUCCEEDED" != "1" ]; then
    echo "[cleanup] run did not complete successfully; skip shutdown."
  else
    case "$SHUTDOWN_ON_EXIT" in
      1|true|TRUE|yes|YES)
        if command -v shutdown >/dev/null 2>&1; then
          echo "[cleanup] scheduling shutdown in ${SHUTDOWN_DELAY_MINUTES} minutes."
          echo "[cleanup] cancel with: shutdown -c"
          shutdown -h +"$SHUTDOWN_DELAY_MINUTES" || true
        else
          echo "[cleanup] shutdown command not found; skip shutdown."
        fi
        ;;
      *)
        echo "[cleanup] SHUTDOWN_ON_EXIT=$SHUTDOWN_ON_EXIT, skip shutdown."
        ;;
    esac
  fi

  exit "$exit_code"
}

trap cleanup EXIT INT TERM

run_case() {
  local subset="$1"
  shift
  local val_tag
  if [ "$VAL_SUBSET" = "0" ] || [ "$VAL_SUBSET" = "all" ]; then
    val_tag="valall"
  else
    val_tag="val${VAL_SUBSET}"
  fi
  local data_dir="$ROOT_DIR/data/modelnet40_train${subset}_${val_tag}"
  local output_dir="$ROOT_DIR/outputs/pointllm-modelnet40-freeform-train${subset}-${val_tag}-r${ROLLOUT_N}-mean1-stable"

  echo
  echo "[run] train_subset=$subset val_subset=$VAL_SUBSET rollout_n=$ROLLOUT_N val_rollout_n=$VAL_ROLLOUT_N test_freq=$TEST_FREQ"
  echo "[run] data_dir=$data_dir"
  echo "[run] output_dir=$output_dir"

  bash "$ROOT_DIR/verl/examples/ttrv/run_3d.sh" \
    SUBSET="$subset" \
    VAL_SUBSET="$VAL_SUBSET" \
    DATA_DIR="$data_dir" \
    NO_GPU="$NO_GPU" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
    ROLLOUT_N="$ROLLOUT_N" \
    VAL_ROLLOUT_N="$VAL_ROLLOUT_N" \
    VAL_DO_SAMPLE="$VAL_DO_SAMPLE" \
    VAL_TEMPERATURE="$VAL_TEMPERATURE" \
    VAL_TOP_P="$VAL_TOP_P" \
    TEST_FREQ="$TEST_FREQ" \
    ACTOR_LR="$ACTOR_LR" \
    USE_KL_LOSS="$USE_KL_LOSS" \
    KL_LOSS_COEF="$KL_LOSS_COEF" \
    LR_WARMUP_STEPS_RATIO="$LR_WARMUP_STEPS_RATIO" \
    REWARD_ALPHA="$REWARD_ALPHA" \
    UNKNOWN_REWARD="$UNKNOWN_REWARD" \
    OUTPUT_DIR="$output_dir" \
    trainer.val_before_train=True \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable \
    "$@"

  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}

echo "[run] 3D TTRV ModelNet40 sweep"
echo "[run] root=$ROOT_DIR"
echo "[run] nohup_log=${NOHUP_LOG:-stdout}"
echo "[run] cases: $CASES"
echo "[run] train rollout_n=$ROLLOUT_N; validation mean@1"
echo "[run] actor_lr=$ACTOR_LR use_kl_loss=$USE_KL_LOSS kl_loss_coef=$KL_LOSS_COEF"
echo "[run] reward_alpha=$REWARD_ALPHA unknown_reward=$UNKNOWN_REWARD"
echo "[run] shutdown_on_exit=$SHUTDOWN_ON_EXIT delay=${SHUTDOWN_DELAY_MINUTES}m"

for case_subset in $CASES; do
  run_case "$case_subset" "$@"
done

echo
echo "[run] completed cases: $CASES"
RUN_SUCCEEDED=1
