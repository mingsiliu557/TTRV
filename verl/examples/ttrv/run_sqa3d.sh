#!/bin/bash
set -euo pipefail
if [ "${TRACE:-0}" = "1" ]; then
  set -x
fi

ORIGINAL_ARGS=("$@")

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
ROOT_DIR=$(cd "$VERL_DIR/.." && pwd)
cd "$ROOT_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
  source "$CONDA_PREFIX/etc/profile.d/conda.sh"
fi
conda activate ttrv

export PYTHONPATH="$VERL_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/.cache}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
unset TRANSFORMERS_CACHE
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/.cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/.cache/triton}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/root/autodl-tmp/.cache/wandb}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
unset VLLM_ATTENTION_BACKEND

RUN_MODE="${RUN_MODE:-adapt}"
DEBUG="${DEBUG:-1}"
SUBSET="${SUBSET:-20}"
ADAPT_SUBSET="${ADAPT_SUBSET:-${TRAIN_SUBSET:-$SUBSET}}"
EVAL_SUBSET="${EVAL_SUBSET:-all}"
SEED="${SEED:-42}"
NO_GPU="${NO_GPU:-$(python -c 'import torch; print(torch.cuda.device_count() or 1)' 2>/dev/null || echo 1)}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$NO_GPU}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$NO_GPU}"
ROLLOUT_N="${ROLLOUT_N:-}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-}"
N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-32}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
TRAIN_TOP_P="${TRAIN_TOP_P:-0.95}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
USE_KL_LOSS="${USE_KL_LOSS:-True}"
VAL_NUM_EXAMINE="${VAL_NUM_EXAMINE:-8}"
PRINT_CONFIG="${PRINT_CONFIG:-False}"
LOG_VALIDATION_BATCHES="${LOG_VALIDATION_BATCHES:-False}"
DUMP_VALIDATION_PREDICTIONS="${DUMP_VALIDATION_PREDICTIONS:-False}"
PRINT_VALIDATION_PREDICTIONS="${PRINT_VALIDATION_PREDICTIONS:-False}"
VALIDATION_PREDICTIONS_DIR="${VALIDATION_PREDICTIONS_DIR:-}"
SAVE_RUN_CONFIG="${SAVE_RUN_CONFIG:-False}"
LOG_TRAIN_GROUPS="${LOG_TRAIN_GROUPS:-}"
LOG_TRAIN_GROUPS_LIMIT="${LOG_TRAIN_GROUPS_LIMIT:-20}"
LOG_TRAIN_RESPONSE_CHARS="${LOG_TRAIN_RESPONSE_CHARS:-160}"
LOG_PARAM_CHECKSUM="${LOG_PARAM_CHECKSUM:-True}"
ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-fp32}"
LL3DA_TRAIN_SCOPE="${LL3DA_TRAIN_SCOPE:-llm}"
REWARD_STRATEGY="${REWARD_STRATEGY:-ttrv_answer_space}"
EM_BONUS="${EM_BONUS:-1.0}"
FREQUENCY_BETA="${FREQUENCY_BETA:-1.0}"
DEBUG_STEPS="${DEBUG_STEPS:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-7524}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
DATA_FILE_PROVIDED="${DATA_FILE+x}"
OUTPUT_DIR_PROVIDED="${OUTPUT_DIR+x}"
RUN_TAG="${RUN_TAG:-}"
MODEL_NAME="${MODEL_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/output}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/data/hf_models/facebook/opt-1.3b}"
LL3DA_REPO_PATH="${LL3DA_REPO_PATH:-$ROOT_DIR/LL3DA}"
LL3DA_CKPT_PATH="${LL3DA_CKPT_PATH:-$ROOT_DIR/data/ll3da_weights/ll3da-opt-1.3b.pth}"
LL3DA_DETECTOR_CKPT_PATH="${LL3DA_DETECTOR_CKPT_PATH:-$ROOT_DIR/LL3DA/pretrained/vote2cap-detr/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth}"
SQA3D_DATA_DIR="${SQA3D_DATA_DIR:-$ROOT_DIR/data/sqa3d_data}"
SCANNET_DIR="${SCANNET_DIR:-$ROOT_DIR/data/scannet/scannet_data}"
DATA_FILE="${DATA_FILE:-$ROOT_DIR/data/sqa3d_test_subset${SUBSET}.parquet}"
ANSWER_DICT_PATH="${ANSWER_DICT_PATH:-$SQA3D_DATA_DIR/answer_dict.json}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.03}"
REWARD_ALPHA="${REWARD_ALPHA:-0.75}"
UNKNOWN_REWARD="${UNKNOWN_REWARD:--1.0}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-}"
TEST_FREQ="${TEST_FREQ:-}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
DATA_FILE_ARG_SET=0
ADAPT_DATA_FILE_ARG_SET=0
EVAL_DATA_FILE_ARG_SET=0
OUTPUT_DIR_ARG_SET=0
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    RUN_MODE=*) RUN_MODE="${arg#RUN_MODE=}" ;;
    DEBUG=*) DEBUG="${arg#DEBUG=}" ;;
    TRACE=*) TRACE="${arg#TRACE=}" ;;
    SUBSET=*) SUBSET="${arg#SUBSET=}"; ADAPT_SUBSET="${arg#SUBSET=}" ;;
    ADAPT_SUBSET=*) ADAPT_SUBSET="${arg#ADAPT_SUBSET=}" ;;
    TRAIN_SUBSET=*) ADAPT_SUBSET="${arg#TRAIN_SUBSET=}" ;;
    EVAL_SUBSET=*) EVAL_SUBSET="${arg#EVAL_SUBSET=}" ;;
    SEED=*) SEED="${arg#SEED=}" ;;
    NO_GPU=*) NO_GPU="${arg#NO_GPU=}" ;;
    TRAIN_BATCH_SIZE=*) TRAIN_BATCH_SIZE="${arg#TRAIN_BATCH_SIZE=}" ;;
    VAL_BATCH_SIZE=*) VAL_BATCH_SIZE="${arg#VAL_BATCH_SIZE=}" ;;
    ROLLOUT_N=*) ROLLOUT_N="${arg#ROLLOUT_N=}" ;;
    N_SAMPLES_PER_PROMPT=*) N_SAMPLES_PER_PROMPT="${arg#N_SAMPLES_PER_PROMPT=}" ;;
    N_VOTES_PER_PROMPT=*) N_VOTES_PER_PROMPT="${arg#N_VOTES_PER_PROMPT=}" ;;
    VAL_ROLLOUT_N=*) VAL_ROLLOUT_N="${arg#VAL_ROLLOUT_N=}" ;;
    TRAIN_TOP_P=*) TRAIN_TOP_P="${arg#TRAIN_TOP_P=}" ;;
    VAL_TOP_P=*) VAL_TOP_P="${arg#VAL_TOP_P=}" ;;
    USE_KL_LOSS=*) USE_KL_LOSS="${arg#USE_KL_LOSS=}" ;;
    VAL_NUM_EXAMINE=*) VAL_NUM_EXAMINE="${arg#VAL_NUM_EXAMINE=}" ;;
    PRINT_CONFIG=*) PRINT_CONFIG="${arg#PRINT_CONFIG=}" ;;
    LOG_VALIDATION_BATCHES=*) LOG_VALIDATION_BATCHES="${arg#LOG_VALIDATION_BATCHES=}" ;;
    DUMP_VALIDATION_PREDICTIONS=*) DUMP_VALIDATION_PREDICTIONS="${arg#DUMP_VALIDATION_PREDICTIONS=}" ;;
    PRINT_VALIDATION_PREDICTIONS=*) PRINT_VALIDATION_PREDICTIONS="${arg#PRINT_VALIDATION_PREDICTIONS=}" ;;
    VALIDATION_PREDICTIONS_DIR=*) VALIDATION_PREDICTIONS_DIR="${arg#VALIDATION_PREDICTIONS_DIR=}" ;;
    SAVE_RUN_CONFIG=*) SAVE_RUN_CONFIG="${arg#SAVE_RUN_CONFIG=}" ;;
    LOG_TRAIN_GROUPS=*) LOG_TRAIN_GROUPS="${arg#LOG_TRAIN_GROUPS=}" ;;
    LOG_TRAIN_GROUPS_LIMIT=*) LOG_TRAIN_GROUPS_LIMIT="${arg#LOG_TRAIN_GROUPS_LIMIT=}" ;;
    LOG_TRAIN_RESPONSE_CHARS=*) LOG_TRAIN_RESPONSE_CHARS="${arg#LOG_TRAIN_RESPONSE_CHARS=}" ;;
    LOG_PARAM_CHECKSUM=*) LOG_PARAM_CHECKSUM="${arg#LOG_PARAM_CHECKSUM=}" ;;
    ACTOR_MODEL_DTYPE=*) ACTOR_MODEL_DTYPE="${arg#ACTOR_MODEL_DTYPE=}" ;;
    LL3DA_TRAIN_SCOPE=*) LL3DA_TRAIN_SCOPE="${arg#LL3DA_TRAIN_SCOPE=}" ;;
    DEBUG_STEPS=*) DEBUG_STEPS="${arg#DEBUG_STEPS=}" ;;
    MAX_PROMPT_LENGTH=*) MAX_PROMPT_LENGTH="${arg#MAX_PROMPT_LENGTH=}" ;;
    MAX_RESPONSE_LENGTH=*) MAX_RESPONSE_LENGTH="${arg#MAX_RESPONSE_LENGTH=}" ;;
    RUN_TAG=*) RUN_TAG="${arg#RUN_TAG=}" ;;
    MODEL_NAME=*) MODEL_NAME="${arg#MODEL_NAME=}" ;;
    OUTPUT_ROOT=*) OUTPUT_ROOT="${arg#OUTPUT_ROOT=}" ;;
    MODEL_PATH=*) MODEL_PATH="${arg#MODEL_PATH=}" ;;
    LL3DA_REPO_PATH=*) LL3DA_REPO_PATH="${arg#LL3DA_REPO_PATH=}" ;;
    LL3DA_CKPT_PATH=*) LL3DA_CKPT_PATH="${arg#LL3DA_CKPT_PATH=}" ;;
    LL3DA_DETECTOR_CKPT_PATH=*) LL3DA_DETECTOR_CKPT_PATH="${arg#LL3DA_DETECTOR_CKPT_PATH=}" ;;
    SQA3D_DATA_DIR=*) SQA3D_DATA_DIR="${arg#SQA3D_DATA_DIR=}" ;;
    SCANNET_DIR=*) SCANNET_DIR="${arg#SCANNET_DIR=}" ;;
    DATA_FILE=*) DATA_FILE="${arg#DATA_FILE=}"; DATA_FILE_ARG_SET=1 ;;
    ADAPT_DATA_FILE=*) ADAPT_DATA_FILE="${arg#ADAPT_DATA_FILE=}"; ADAPT_DATA_FILE_ARG_SET=1 ;;
    EVAL_DATA_FILE=*) EVAL_DATA_FILE="${arg#EVAL_DATA_FILE=}"; EVAL_DATA_FILE_ARG_SET=1 ;;
    ANSWER_DICT_PATH=*) ANSWER_DICT_PATH="${arg#ANSWER_DICT_PATH=}" ;;
    OUTPUT_DIR=*) OUTPUT_DIR="${arg#OUTPUT_DIR=}"; OUTPUT_DIR_ARG_SET=1 ;;
    ACTOR_LR=*) ACTOR_LR="${arg#ACTOR_LR=}" ;;
    LR_WARMUP_STEPS_RATIO=*) LR_WARMUP_STEPS_RATIO="${arg#LR_WARMUP_STEPS_RATIO=}" ;;
    REWARD_ALPHA=*) REWARD_ALPHA="${arg#REWARD_ALPHA=}" ;;
    UNKNOWN_REWARD=*) UNKNOWN_REWARD="${arg#UNKNOWN_REWARD=}" ;;
    REWARD_STRATEGY=*) REWARD_STRATEGY="${arg#REWARD_STRATEGY=}" ;;
    EM_BONUS=*) EM_BONUS="${arg#EM_BONUS=}" ;;
    FREQUENCY_BETA=*) FREQUENCY_BETA="${arg#FREQUENCY_BETA=}" ;;
    VAL_BEFORE_TRAIN=*) VAL_BEFORE_TRAIN="${arg#VAL_BEFORE_TRAIN=}" ;;
    TEST_FREQ=*) TEST_FREQ="${arg#TEST_FREQ=}" ;;
    SAVE_FREQ=*) SAVE_FREQ="${arg#SAVE_FREQ=}" ;;
    TOTAL_EPOCHS=*) TOTAL_EPOCHS="${arg#TOTAL_EPOCHS=}" ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

if [ "${TRACE:-0}" = "1" ]; then
  set -x
fi

RUN_MODE="$(echo "$RUN_MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$RUN_MODE" != "baseline" ] && [ "$RUN_MODE" != "adapt" ]; then
  echo "RUN_MODE must be baseline or adapt, got: $RUN_MODE" >&2
  exit 1
fi
if [ -z "$VAL_BEFORE_TRAIN" ]; then
  VAL_BEFORE_TRAIN=True
fi
if [ -z "$TEST_FREQ" ]; then
  if [ "$RUN_MODE" = "baseline" ]; then
    TEST_FREQ=1
  else
    TEST_FREQ=1
  fi
fi

if [ "$DATA_FILE_ARG_SET" = "1" ] || [ -n "$DATA_FILE_PROVIDED" ]; then
  ADAPT_DATA_FILE="${ADAPT_DATA_FILE:-$DATA_FILE}"
  EVAL_DATA_FILE="${EVAL_DATA_FILE:-$DATA_FILE}"
fi
if [ "$ADAPT_DATA_FILE_ARG_SET" = "0" ] && [ -z "${ADAPT_DATA_FILE+x}" ]; then
  ADAPT_DATA_FILE="$ROOT_DIR/data/sqa3d_adapt_subset${ADAPT_SUBSET}.parquet"
fi
if [ "$EVAL_DATA_FILE_ARG_SET" = "0" ] && [ -z "${EVAL_DATA_FILE+x}" ]; then
  if [ "$EVAL_SUBSET" = "all" ] || [ "$EVAL_SUBSET" = "full" ]; then
    EVAL_DATA_FILE="$ROOT_DIR/data/sqa3d_test_all.parquet"
  else
    EVAL_DATA_FILE="$ROOT_DIR/data/sqa3d_test_subset${EVAL_SUBSET}.parquet"
  fi
fi
if [ -z "$MODEL_NAME" ]; then
  MODEL_NAME="$(basename "$LL3DA_CKPT_PATH" .pth)"
fi
MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/: ' '___')

if [ -z "$ROLLOUT_N" ]; then
  if [ -n "$N_SAMPLES_PER_PROMPT" ]; then
    ROLLOUT_N="$N_SAMPLES_PER_PROMPT"
  elif [ "$DEBUG" = "1" ]; then
    ROLLOUT_N=2
  else
    ROLLOUT_N=32
  fi
fi
if [ -z "$N_SAMPLES_PER_PROMPT" ]; then
  N_SAMPLES_PER_PROMPT="$ROLLOUT_N"
fi
if [ -z "$DEBUG_STEPS" ] && [ "$DEBUG" = "1" ]; then
  DEBUG_STEPS=1
fi
if [ -z "$LOG_TRAIN_GROUPS" ]; then
  if [ "$DEBUG" = "1" ]; then
    LOG_TRAIN_GROUPS=True
  else
    LOG_TRAIN_GROUPS=False
  fi
fi

tag_value() {
  printf '%s' "$1" | sed -e 's/+//g' -e 's/-/m/g' -e 's/\./p/g' -e 's/[^A-Za-z0-9_]/_/g'
}

if [ -z "$RUN_TAG" ]; then
  REWARD_STRATEGY_TAG="$REWARD_STRATEGY"
  if [ "$REWARD_STRATEGY_TAG" = "ttrv_answer_space" ]; then
    REWARD_STRATEGY_TAG="ttrv"
  fi
  RUN_TAG="$(date +%Y%m%d_%H%M%S)-${RUN_MODE}${ADAPT_SUBSET}-bs${TRAIN_BATCH_SIZE}-ep${TOTAL_EPOCHS}-r${ROLLOUT_N}-s${N_SAMPLES_PER_PROMPT}-v${N_VOTES_PER_PROMPT}-tp$(tag_value "$TRAIN_TOP_P")-a$(tag_value "$REWARD_ALPHA")-lr$(tag_value "$ACTOR_LR")-mrl${MAX_RESPONSE_LENGTH}-rs$(tag_value "$REWARD_STRATEGY_TAG")"
  if [ "$LL3DA_TRAIN_SCOPE" != "llm" ]; then
    RUN_TAG="${RUN_TAG}-scope$(tag_value "$LL3DA_TRAIN_SCOPE")"
  fi
fi

if [ "$OUTPUT_DIR_ARG_SET" = "0" ] && [ -z "$OUTPUT_DIR_PROVIDED" ]; then
  OUTPUT_DIR="$OUTPUT_ROOT/$MODEL_NAME_SAFE/$RUN_TAG"
fi
if [ -z "$VALIDATION_PREDICTIONS_DIR" ]; then
  VALIDATION_PREDICTIONS_DIR="$OUTPUT_DIR"
fi

hf_download_file() {
  local repo_id="$1"
  local filename="$2"
  local local_dir="$3"
  python - "$repo_id" "$filename" "$local_dir" <<'PY'
import sys
from huggingface_hub import hf_hub_download

repo_id, filename, local_dir = sys.argv[1:4]
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(path)
PY
}

hf_snapshot() {
  local repo_id="$1"
  local local_dir="$2"
  python - "$repo_id" "$local_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1:3]
path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(path)
PY
}

if [ ! -d "$LL3DA_REPO_PATH" ]; then
  git clone https://github.com/Open3DA/LL3DA.git "$LL3DA_REPO_PATH"
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
  mkdir -p "$MODEL_PATH"
  for file in \
    config.json \
    generation_config.json \
    merges.txt \
    pytorch_model.bin \
    special_tokens_map.json \
    tokenizer_config.json \
    vocab.json; do
    hf_download_file facebook/opt-1.3b "$file" "$MODEL_PATH"
  done
fi

if [ ! -f "$LL3DA_REPO_PATH/bert-base-embedding/config.json" ]; then
  mkdir -p "$LL3DA_REPO_PATH/bert-base-embedding"
  hf_snapshot CH3COOK/bert-base-embedding "$LL3DA_REPO_PATH/bert-base-embedding"
fi

if [ ! -f "$LL3DA_CKPT_PATH" ]; then
  mkdir -p "$(dirname "$LL3DA_CKPT_PATH")"
  hf_download_file CH3COOK/LL3DA-weight-release ll3da-opt-1.3b.pth "$(dirname "$LL3DA_CKPT_PATH")"
fi

if [ ! -f "$SQA3D_DATA_DIR/v1_balanced_questions_test_scannetv2.json" ] || [ ! -f "$ANSWER_DICT_PATH" ]; then
  SQA3D_ZIP="$ROOT_DIR/data/sqa3d_data/sqa_task.zip"
  SQA3D_EXTRACT="$ROOT_DIR/data/sqa3d_data/_sqa_task_extract"
  mkdir -p "$SQA3D_DATA_DIR" "$SQA3D_EXTRACT"
  wget -nc -O "$SQA3D_ZIP" "https://zenodo.org/records/7792397/files/sqa_task.zip?download=1"
  unzip -n "$SQA3D_ZIP" -d "$SQA3D_EXTRACT"
  cp "$(find "$SQA3D_EXTRACT" -name answer_dict.json | head -1)" "$SQA3D_DATA_DIR/answer_dict.json"
  for split in train val test; do
    cp "$(find "$SQA3D_EXTRACT" -name "v1_balanced_questions_${split}_scannetv2.json" | head -1)" \
      "$SQA3D_DATA_DIR/v1_balanced_questions_${split}_scannetv2.json"
    cp "$(find "$SQA3D_EXTRACT" -name "v1_balanced_sqa_annotations_${split}_scannetv2.json" | head -1)" \
      "$SQA3D_DATA_DIR/v1_balanced_sqa_annotations_${split}_scannetv2.json"
  done
fi

if [ ! -d "$SCANNET_DIR" ]; then
  mkdir -p "$ROOT_DIR/data/scannet"
  hf_download_file CH3COOK/LL3DA-weight-release scannet_data.zip "$ROOT_DIR/data/scannet"
  unzip -n "$ROOT_DIR/data/scannet/scannet_data.zip" -d "$ROOT_DIR/data/scannet"
fi

prepare_sqa3d_file() {
  local output_file="$1"
  local subset_value="$2"
  if [ -f "$output_file" ]; then
    if python - "$output_file" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
try:
    df = pd.read_parquet(path, columns=["extra_info"])
    if len(df) == 0:
        sys.exit(1)
    version = (df.iloc[0]["extra_info"] or {}).get("prompt_version")
    sys.exit(0 if version == "sqa3d_short_answer_v3_ll3da_aligned" else 1)
except Exception:
    sys.exit(1)
PY
    then
      return
    fi
    echo "Regenerating $output_file for SQA3D prompt_version=sqa3d_short_answer_v3_ll3da_aligned"
  fi
  python "$ROOT_DIR/scripts_3d/prepare_sqa3d_subset.py" \
    --data-dir "$SQA3D_DATA_DIR" \
    --scannet-dir "$SCANNET_DIR" \
    --output-file "$output_file" \
    --subset "$subset_value" \
    --seed "$SEED"
}

if [ "$RUN_MODE" = "adapt" ] || [ ! -f "$ADAPT_DATA_FILE" ]; then
  prepare_sqa3d_file "$ADAPT_DATA_FILE" "$ADAPT_SUBSET"
fi
prepare_sqa3d_file "$EVAL_DATA_FILE" "$EVAL_SUBSET"

mkdir -p "$OUTPUT_DIR"

RUN_ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
  data.train_files=["$ADAPT_DATA_FILE"]
  data.val_files=["$EVAL_DATA_FILE"]
  data.cache_dir=/root/autodl-tmp/.cache/verl/rlhf
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.val_batch_size="$VAL_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.point_cloud_num_points=40000
  data.shuffle=False
  data.filter_overlong_prompts=False
  data.truncation=error
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.model.lora_rank=0
  actor_rollout_ref.model.ll3da_repo_path="$LL3DA_REPO_PATH"
  actor_rollout_ref.model.ll3da_ckpt_path="$LL3DA_CKPT_PATH"
  actor_rollout_ref.model.ll3da_detector_ckpt_path="$LL3DA_DETECTOR_CKPT_PATH"
  actor_rollout_ref.model.ll3da_qformer_vocab=bert-base-embedding
  actor_rollout_ref.model.ll3da_use_color=True
  actor_rollout_ref.model.ll3da_use_normal=True
  actor_rollout_ref.model.ll3da_no_height=False
  actor_rollout_ref.model.ll3da_train_scope="$LL3DA_TRAIN_SCOPE"
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS"
  +actor_rollout_ref.actor.log_param_checksum="$LOG_PARAM_CHECKSUM"
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO"
  actor_rollout_ref.actor.optim.warmup_style=cosine
  actor_rollout_ref.actor.optim.weight_decay=0.01
  actor_rollout_ref.actor.fsdp_config.model_dtype="$ACTOR_MODEL_DTYPE"
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.rollout.name=hf
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p="$TRAIN_TOP_P"
  actor_rollout_ref.rollout.top_k=0
  actor_rollout_ref.rollout.do_sample=True
  actor_rollout_ref.rollout.do_vote=True
  actor_rollout_ref.rollout.n_vote="$N_VOTES_PER_PROMPT"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.val_kwargs.do_sample=False
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_ROLLOUT_N"
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P"
  actor_rollout_ref.rollout.val_kwargs.top_k=0
  reward_model.reward_manager=batch
  +reward_model.val_num_examine="$VAL_NUM_EXAMINE"
  reward_model.reward_kwargs.alpha="$REWARD_ALPHA"
  +reward_model.reward_kwargs.reward_strategy="$REWARD_STRATEGY"
  +reward_model.reward_kwargs.em_bonus="$EM_BONUS"
  +reward_model.reward_kwargs.frequency_beta="$FREQUENCY_BETA"
  +reward_model.reward_kwargs.unknown_reward="$UNKNOWN_REWARD"
  +reward_model.reward_kwargs.answer_dict_path="$ANSWER_DICT_PATH"
  +reward_model.reward_kwargs.train_group_size="$N_VOTES_PER_PROMPT"
  +reward_model.reward_kwargs.log_train_groups="$LOG_TRAIN_GROUPS"
  +reward_model.reward_kwargs.log_train_groups_limit="$LOG_TRAIN_GROUPS_LIMIT"
  +reward_model.reward_kwargs.log_train_response_chars="$LOG_TRAIN_RESPONSE_CHARS"
  reward_model.reward_kwargs.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT"
  reward_model.reward_kwargs.n_votes_per_prompt="$N_VOTES_PER_PROMPT"
  custom_reward_function.path="$VERL_DIR/verl/utils/reward_score/sqa3d_ttrv.py"
  custom_reward_function.name=compute_score
  trainer.logger=['console']
  +trainer.print_config="$PRINT_CONFIG"
  +trainer.log_validation_batches="$LOG_VALIDATION_BATCHES"
  +trainer.dump_validation_predictions="$DUMP_VALIDATION_PREDICTIONS"
  +trainer.print_validation_predictions="$PRINT_VALIDATION_PREDICTIONS"
  +trainer.validation_predictions_dir="$VALIDATION_PREDICTIONS_DIR"
  trainer.project_name=ttrv-3d
  trainer.experiment_name=ll3da-sqa3d-"$RUN_MODE"-adapt"$ADAPT_SUBSET"-eval"$EVAL_SUBSET"
  trainer.n_gpus_per_node="$NO_GPU"
  trainer.nnodes=1
  trainer.val_before_train="$VAL_BEFORE_TRAIN"
  trainer.test_freq="$TEST_FREQ"
  trainer.save_freq="$SAVE_FREQ"
  trainer.total_epochs="$TOTAL_EPOCHS"
  trainer.default_local_dir="$OUTPUT_DIR/checkpoints"
)

if [ "$RUN_MODE" = "baseline" ]; then
  RUN_ARGS+=(+trainer.val_only=True)
fi

if [ -n "$DEBUG_STEPS" ]; then
  RUN_ARGS+=(trainer.total_training_steps="$DEBUG_STEPS")
fi

COMMAND_FILE="$OUTPUT_DIR/command.sh"
EXPANDED_COMMAND_FILE="$OUTPUT_DIR/command_expanded.txt"
RUN_CONFIG_FILE="$OUTPUT_DIR/run_config.txt"
LOG_FILE="$OUTPUT_DIR/run.log"

{
  printf '#!/bin/bash\n'
  printf 'set -euo pipefail\n'
  printf 'cd %q\n' "$ROOT_DIR"
  printf 'bash %q' "$SCRIPT_DIR/run_sqa3d.sh"
  for arg in "${ORIGINAL_ARGS[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
} > "$COMMAND_FILE"
chmod +x "$COMMAND_FILE"

if [ "$SAVE_RUN_CONFIG" = "True" ]; then
  {
    printf 'python -m verl.trainer.main_ppo'
    for arg in "${RUN_ARGS[@]}" "${EXTRA_ARGS[@]}"; do
      printf ' %q' "$arg"
    done
    printf '\n'
  } > "$EXPANDED_COMMAND_FILE"

  {
    printf 'run_mode=%s\n' "$RUN_MODE"
    printf 'model_name=%s\n' "$MODEL_NAME_SAFE"
    printf 'run_tag=%s\n' "$RUN_TAG"
    printf 'output_dir=%s\n' "$OUTPUT_DIR"
    printf 'adapt_data_file=%s\n' "$ADAPT_DATA_FILE"
    printf 'eval_data_file=%s\n' "$EVAL_DATA_FILE"
    printf 'no_gpu=%s\n' "$NO_GPU"
    printf 'train_batch_size=%s\n' "$TRAIN_BATCH_SIZE"
    printf 'val_batch_size=%s\n' "$VAL_BATCH_SIZE"
    printf 'rollout_n=%s\n' "$ROLLOUT_N"
    printf 'n_samples_per_prompt=%s\n' "$N_SAMPLES_PER_PROMPT"
    printf 'n_votes_per_prompt=%s\n' "$N_VOTES_PER_PROMPT"
    printf 'val_rollout_n=%s\n' "$VAL_ROLLOUT_N"
    printf 'val_before_train=%s\n' "$VAL_BEFORE_TRAIN"
    printf 'test_freq=%s\n' "$TEST_FREQ"
    printf 'actor_lr=%s\n' "$ACTOR_LR"
    printf 'use_kl_loss=%s\n' "$USE_KL_LOSS"
    printf 'train_top_p=%s\n' "$TRAIN_TOP_P"
    printf 'val_top_p=%s\n' "$VAL_TOP_P"
    printf 'val_num_examine=%s\n' "$VAL_NUM_EXAMINE"
    printf 'max_prompt_length=%s\n' "$MAX_PROMPT_LENGTH"
    printf 'max_response_length=%s\n' "$MAX_RESPONSE_LENGTH"
    printf 'reward_alpha=%s\n' "$REWARD_ALPHA"
    printf 'reward_strategy=%s\n' "$REWARD_STRATEGY"
    printf 'primary_eval_metric=%s\n' "official_sqa3d_freeform_em"
    printf 'ttrv_frequency_space=%s\n' "official_sqa3d_answer_dict_extraction"
    printf 'ttrv_entropy=%s\n' "normalized_answer_distribution_entropy"
    printf 'll3da_point_cloud=%s\n' "aligned_xyz_rgb_normal_height_10d"
    printf 'll3da_interact_placeholder=%s\n' "empty_box_and_click_queries"
    printf 'em_bonus=%s\n' "$EM_BONUS"
    printf 'frequency_beta=%s\n' "$FREQUENCY_BETA"
    printf 'unknown_reward=%s\n' "$UNKNOWN_REWARD"
    printf 'log_train_groups=%s\n' "$LOG_TRAIN_GROUPS"
    printf 'log_train_groups_limit=%s\n' "$LOG_TRAIN_GROUPS_LIMIT"
    printf 'log_train_response_chars=%s\n' "$LOG_TRAIN_RESPONSE_CHARS"
    printf 'log_param_checksum=%s\n' "$LOG_PARAM_CHECKSUM"
    printf 'actor_model_dtype=%s\n' "$ACTOR_MODEL_DTYPE"
    printf 'll3da_train_scope=%s\n' "$LL3DA_TRAIN_SCOPE"
    printf 'print_config=%s\n' "$PRINT_CONFIG"
    printf 'log_validation_batches=%s\n' "$LOG_VALIDATION_BATCHES"
    printf 'dump_validation_predictions=%s\n' "$DUMP_VALIDATION_PREDICTIONS"
    printf 'print_validation_predictions=%s\n' "$PRINT_VALIDATION_PREDICTIONS"
    printf 'validation_predictions_dir=%s\n' "$VALIDATION_PREDICTIONS_DIR"
  } > "$RUN_CONFIG_FILE"
fi

set +e
python -m verl.trainer.main_ppo "${RUN_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
RUN_STATUS=${PIPESTATUS[0]}
set -e

python "$ROOT_DIR/scripts_3d/summarize_sqa3d_results.py" \
  --log-file "$LOG_FILE" \
  --output-file "$OUTPUT_DIR/results.md" \
  --title "LL3DA SQA3D ${RUN_MODE} adapt=${ADAPT_SUBSET} eval=${EVAL_SUBSET}" || true

python "$ROOT_DIR/scripts_3d/diagnose_sqa3d_unknown.py" \
  --parquet-file "$EVAL_DATA_FILE" \
  --answer-dict-path "$ANSWER_DICT_PATH" \
  --log-file "$LOG_FILE" | tee "$OUTPUT_DIR/unknown_diagnosis.txt" || true

exit "$RUN_STATUS"
