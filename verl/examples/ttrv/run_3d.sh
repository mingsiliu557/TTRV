#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
ROOT_DIR=$(cd "$VERL_DIR/.." && pwd)
cd "$ROOT_DIR"

source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh" || source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ttrv

export PYTHONPATH="$VERL_DIR:$ROOT_DIR/pointllm-src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/.cache}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
unset TRANSFORMERS_CACHE
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/.cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/root/autodl-tmp/.cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/.cache/triton}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/root/autodl-tmp/.cache/wandb}"
unset VLLM_ATTENTION_BACKEND

if ! python -c "import pointllm" >/dev/null 2>&1; then
  if [ ! -d "$ROOT_DIR/pointllm-src" ]; then
    git clone https://github.com/OpenRobotLab/PointLLM.git "$ROOT_DIR/pointllm-src"
  fi
  pip install -e "$ROOT_DIR/pointllm-src" --no-deps
fi

SUBSET="${SUBSET:-20}"
VAL_SUBSET="${VAL_SUBSET:-0}"
SEED="${SEED:-0}"
DATA_DIR="${DATA_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
NO_GPU="${NO_GPU:-$(python -c 'import torch; print(torch.cuda.device_count() or 1)' 2>/dev/null || echo 1)}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-20}"
ROLLOUT_N="${ROLLOUT_N:-20}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
VAL_TOP_K="${VAL_TOP_K:--1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-561}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-24}"
MODEL_PATH="${MODEL_PATH:-RunsenXu/PointLLM_7B_v1.2}"
DEBUG_STEPS="${DEBUG_STEPS:-}"
TEST_FREQ="${TEST_FREQ:-}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
USE_KL_LOSS="${USE_KL_LOSS:-True}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.03}"
REWARD_ALPHA="${REWARD_ALPHA:-0.75}"
UNKNOWN_REWARD="${UNKNOWN_REWARD:--1.0}"
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    DEBUG_STEPS=*) DEBUG_STEPS="${arg#DEBUG_STEPS=}" ;;
    SUBSET=*) SUBSET="${arg#SUBSET=}" ;;
    VAL_SUBSET=*) VAL_SUBSET="${arg#VAL_SUBSET=}" ;;
    SEED=*) SEED="${arg#SEED=}" ;;
    DATA_DIR=*) DATA_DIR="${arg#DATA_DIR=}" ;;
    OUTPUT_DIR=*) OUTPUT_DIR="${arg#OUTPUT_DIR=}" ;;
    NO_GPU=*) NO_GPU="${arg#NO_GPU=}" ;;
    TRAIN_BATCH_SIZE=*) TRAIN_BATCH_SIZE="${arg#TRAIN_BATCH_SIZE=}" ;;
    VAL_BATCH_SIZE=*) VAL_BATCH_SIZE="${arg#VAL_BATCH_SIZE=}" ;;
    ROLLOUT_N=*) ROLLOUT_N="${arg#ROLLOUT_N=}" ;;
    VAL_ROLLOUT_N=*) VAL_ROLLOUT_N="${arg#VAL_ROLLOUT_N=}" ;;
    VAL_DO_SAMPLE=*) VAL_DO_SAMPLE="${arg#VAL_DO_SAMPLE=}" ;;
    VAL_TEMPERATURE=*) VAL_TEMPERATURE="${arg#VAL_TEMPERATURE=}" ;;
    VAL_TOP_P=*) VAL_TOP_P="${arg#VAL_TOP_P=}" ;;
    VAL_TOP_K=*) VAL_TOP_K="${arg#VAL_TOP_K=}" ;;
    MAX_PROMPT_LENGTH=*) MAX_PROMPT_LENGTH="${arg#MAX_PROMPT_LENGTH=}" ;;
    MAX_RESPONSE_LENGTH=*) MAX_RESPONSE_LENGTH="${arg#MAX_RESPONSE_LENGTH=}" ;;
    MODEL_PATH=*) MODEL_PATH="${arg#MODEL_PATH=}" ;;
    TEST_FREQ=*) TEST_FREQ="${arg#TEST_FREQ=}" ;;
    FORCE_PREPARE=*) FORCE_PREPARE="${arg#FORCE_PREPARE=}" ;;
    ACTOR_LR=*) ACTOR_LR="${arg#ACTOR_LR=}" ;;
    USE_KL_LOSS=*) USE_KL_LOSS="${arg#USE_KL_LOSS=}" ;;
    KL_LOSS_COEF=*) KL_LOSS_COEF="${arg#KL_LOSS_COEF=}" ;;
    LR_WARMUP_STEPS_RATIO=*) LR_WARMUP_STEPS_RATIO="${arg#LR_WARMUP_STEPS_RATIO=}" ;;
    REWARD_ALPHA=*) REWARD_ALPHA="${arg#REWARD_ALPHA=}" ;;
    UNKNOWN_REWARD=*) UNKNOWN_REWARD="${arg#UNKNOWN_REWARD=}" ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

if [ -z "$DATA_DIR" ]; then
  if [ "$VAL_SUBSET" = "0" ] || [ "$VAL_SUBSET" = "all" ]; then
    DATA_DIR="$ROOT_DIR/data/modelnet40_train${SUBSET}_valall"
  else
    DATA_DIR="$ROOT_DIR/data/modelnet40_train${SUBSET}_val${VAL_SUBSET}"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/pointllm-modelnet40-freeform-${SUBSET}}"
if [ -z "$TRAIN_BATCH_SIZE" ]; then
  if [ "$NO_GPU" -gt 1 ]; then
    TRAIN_BATCH_SIZE="$NO_GPU"
  else
    TRAIN_BATCH_SIZE=2
  fi
fi

if (( TRAIN_BATCH_SIZE % NO_GPU != 0 )); then
  echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE must be divisible by NO_GPU=$NO_GPU for DP rollout." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"
EXPECTED_VAL_ROWS="$VAL_SUBSET"
if [ "$VAL_SUBSET" = "0" ] || [ "$VAL_SUBSET" = "all" ]; then
  EXPECTED_VAL_ROWS=2468
fi

NEED_PREPARE=0
if [ "$FORCE_PREPARE" = "1" ] || [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
  NEED_PREPARE=1
elif ! python - "$DATA_DIR/train.parquet" "$DATA_DIR/test.parquet" "$SUBSET" "$EXPECTED_VAL_ROWS" <<'PY'
import sys
from datasets import load_dataset

train_file, test_file, expected_train, expected_val = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
train_n = len(load_dataset("parquet", data_files=train_file, split="train"))
test_n = len(load_dataset("parquet", data_files=test_file, split="train"))
if train_n != expected_train or test_n != expected_val:
    print(
        f"Existing ModelNet40 parquet row counts mismatch: "
        f"train={train_n} expected={expected_train}, test={test_n} expected={expected_val}",
        file=sys.stderr,
    )
    sys.exit(1)
PY
then
  NEED_PREPARE=1
fi

if [ "$NEED_PREPARE" = "1" ]; then
  PREPARE_ARGS=(--subset "$SUBSET" --seed "$SEED" --output-dir "$DATA_DIR")
  if [ "$VAL_SUBSET" != "0" ] && [ "$VAL_SUBSET" != "all" ]; then
    PREPARE_ARGS+=(--val-subset "$VAL_SUBSET")
  fi
  python "$ROOT_DIR/scripts_3d/prepare_modelnet40.py" "${PREPARE_ARGS[@]}"
fi

if [ -z "$TEST_FREQ" ]; then
  if [ -n "$DEBUG_STEPS" ]; then
    TEST_FREQ=1
  else
    TEST_FREQ=25
  fi
fi

RUN_ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
  data.train_files=["$DATA_DIR/train.parquet"]
  data.val_files=["$DATA_DIR/test.parquet"]
  data.cache_dir=/root/autodl-tmp/.cache/verl/rlhf
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.val_batch_size="$VAL_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.shuffle=False
  data.filter_overlong_prompts=False
  data.truncation=error
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.external_lib=pointllm.model
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.model.lora_rank=0
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.target_modules='[q_proj,k_proj,v_proj,o_proj]'
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS"
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.ppo_mini_batch_size=2
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO"
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.rollout.name=hf
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=0.95
  actor_rollout_ref.rollout.top_k=0
  actor_rollout_ref.rollout.do_sample=True
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE"
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_ROLLOUT_N"
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE"
  actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P"
  actor_rollout_ref.rollout.val_kwargs.top_k="$VAL_TOP_K"
  reward_model.reward_manager=batch
  reward_model.reward_kwargs.alpha="$REWARD_ALPHA"
  +reward_model.reward_kwargs.unknown_reward="$UNKNOWN_REWARD"
  reward_model.reward_kwargs.n_samples_per_prompt="$ROLLOUT_N"
  reward_model.reward_kwargs.n_votes_per_prompt="$ROLLOUT_N"
  custom_reward_function.path="$VERL_DIR/verl/utils/reward_score/modelnet40_freeform.py"
  custom_reward_function.name=compute_score
  trainer.logger=['console']
  trainer.project_name=ttrv-3d
  trainer.experiment_name=pointllm-modelnet40-freeform-"$SUBSET"
  trainer.n_gpus_per_node="$NO_GPU"
  trainer.nnodes=1
  trainer.val_before_train=True
  trainer.test_freq="$TEST_FREQ"
  trainer.save_freq=-1
  trainer.total_epochs=1
  trainer.default_local_dir="$OUTPUT_DIR/checkpoints"
)

if [ -n "$DEBUG_STEPS" ]; then
  RUN_ARGS+=(trainer.total_training_steps="$DEBUG_STEPS")
fi

python -m verl.trainer.main_ppo "${RUN_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/run_$(date +%m%d_%H%M%S).log"
