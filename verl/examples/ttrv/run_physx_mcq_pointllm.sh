#!/bin/bash
set -euo pipefail
if [ "${TRACE:-0}" = "1" ]; then
  set -x
fi

ORIGINAL_ARGS=("$@")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
ROOT_DIR=$(cd "$VERL_DIR/.." && pwd)
POINTLLM_DIR="$ROOT_DIR/pointllm-src"
cd "$ROOT_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
  source "$CONDA_PREFIX/etc/profile.d/conda.sh"
fi
conda activate ttrv

export PYTHONPATH="$VERL_DIR:$POINTLLM_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/.cache}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
unset TRANSFORMERS_CACHE
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/.cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/.cache/triton}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
unset VLLM_ATTENTION_BACKEND

resolve_pointllm_model() {
  python - <<'PY'
from pathlib import Path
candidates = [
    Path('/root/.cache/huggingface/hub/models--RunsenXu--PointLLM_7B_v1.2'),
    Path('/root/autodl-tmp/.cache/huggingface/hub/models--RunsenXu--PointLLM_7B_v1.2'),
    Path('/root/autodl-tmp/.cache/huggingface/transformers/models--RunsenXu--PointLLM_7B_v1.2'),
]
for candidate in candidates:
    if (candidate / 'config.json').exists():
        print(candidate)
        raise SystemExit(0)
    refs_main = candidate / 'refs' / 'main'
    snapshots = candidate / 'snapshots'
    names = []
    if refs_main.exists():
        names.append(refs_main.read_text().strip())
    if snapshots.exists():
        names.extend(sorted(p.name for p in snapshots.iterdir() if p.is_dir()))
    for name in names:
        snapshot = snapshots / name
        if (snapshot / 'config.json').exists():
            print(snapshot)
            raise SystemExit(0)
raise SystemExit('No local PointLLM_7B_v1.2 config.json found in known cache roots')
PY
}

tag_value() {
  printf '%s' "$1" | sed -e 's/+//g' -e 's/-/m/g' -e 's/\./p/g' -e 's/[^A-Za-z0-9_]/_/g'
}

RUN_MODE="${RUN_MODE:-adapt}"
DEBUG="${DEBUG:-1}"
SUBSET="${SUBSET:-20}"
VAL_SUBSET="${VAL_SUBSET:-20}"
SEED="${SEED:-0}"
NO_GPU="${NO_GPU:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-}"
N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
TRAIN_TOP_P="${TRAIN_TOP_P:-0.95}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
ACTOR_FSDP_MP_DTYPE="${ACTOR_FSDP_MP_DTYPE:-}"
SELECTION="${SELECTION:-first}"
DATA_SHUFFLE="${DATA_SHUFFLE:-False}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-0}"
ACTOR_LORA_ALPHA="${ACTOR_LORA_ALPHA:-16}"
ACTOR_LORA_TARGET_MODULES="${ACTOR_LORA_TARGET_MODULES:-all-linear}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
USE_KL_LOSS="${USE_KL_LOSS:-True}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.03}"
REWARD_ALPHA="${REWARD_ALPHA:-0.75}"
FREQUENCY_BETA="${FREQUENCY_BETA:-1.0}"
UNKNOWN_REWARD="${UNKNOWN_REWARD:--1.0}"
REWARD_VARIANT="${REWARD_VARIANT:-standard}"
TTRL_MIN_MAJORITY_RATIO="${TTRL_MIN_MAJORITY_RATIO:-0.0}"
TTRL_MAX_MAJORITY_RATIO="${TTRL_MAX_MAJORITY_RATIO:-1.0}"
GEO_NUM_VIEWS="${GEO_NUM_VIEWS:-1}"
GEO_SAMPLES_PER_VIEW="${GEO_SAMPLES_PER_VIEW:-0}"
GEO_MIN_VIEW_SUPPORT="${GEO_MIN_VIEW_SUPPORT:-0}"
GEO_MIN_HM="${GEO_MIN_HM:-0.0}"
GEO_MIN_VIEW_PROB="${GEO_MIN_VIEW_PROB:-0.0}"
GEO_MIN_SCORE_MARGIN="${GEO_MIN_SCORE_MARGIN:-0.0}"
GEO_SKIP_QUESTION_TYPES="${GEO_SKIP_QUESTION_TYPES:-}"
GEO_REQUIRE_ORIGINAL_MAJORITY="${GEO_REQUIRE_ORIGINAL_MAJORITY:-False}"
GEO_MIN_ORIGINAL_MAJORITY_RATIO="${GEO_MIN_ORIGINAL_MAJORITY_RATIO:-0.0}"
GEO_SKIP_AMBIGUOUS_JOINT_OPTIONS="${GEO_SKIP_AMBIGUOUS_JOINT_OPTIONS:-False}"
GEO_SOFT_GAMMA="${GEO_SOFT_GAMMA:-2.0}"
GEO_SOFT_MIN_MAX_PROB="${GEO_SOFT_MIN_MAX_PROB:-0.0}"
GEO_SOFT_MIN_KNOWN_COUNT="${GEO_SOFT_MIN_KNOWN_COUNT:-0}"
POINT_REFRAME_POLICY="${POINT_REFRAME_POLICY:-none}"
POINT_REFRAME_NUM_VIEWS="${POINT_REFRAME_NUM_VIEWS:-$GEO_NUM_VIEWS}"
POINT_REFRAME_SAMPLES_PER_VIEW="${POINT_REFRAME_SAMPLES_PER_VIEW:-$GEO_SAMPLES_PER_VIEW}"
POINT_REFRAME_SEED="${POINT_REFRAME_SEED:-$SEED}"
POINT_REFRAME_SCALE_MIN="${POINT_REFRAME_SCALE_MIN:-0.95}"
POINT_REFRAME_SCALE_MAX="${POINT_REFRAME_SCALE_MAX:-1.05}"
POINT_REFRAME_TRANSLATE="${POINT_REFRAME_TRANSLATE:-0.03}"
POINT_REFRAME_JITTER_SIGMA="${POINT_REFRAME_JITTER_SIGMA:-0.01}"
POINT_REFRAME_JITTER_CLIP="${POINT_REFRAME_JITTER_CLIP:-0.03}"
POINT_REFRAME_DOWNSAMPLE_MIN="${POINT_REFRAME_DOWNSAMPLE_MIN:-0.60}"
POINT_REFRAME_DOWNSAMPLE_MAX="${POINT_REFRAME_DOWNSAMPLE_MAX:-0.85}"
POINT_REFRAME_RENORMALIZE="${POINT_REFRAME_RENORMALIZE:-False}"
PHYSX_EARLY_STOP="${PHYSX_EARLY_STOP:-False}"
EARLY_STOP_ACC_DROP="${EARLY_STOP_ACC_DROP:-0.05}"
EARLY_STOP_INVALID_INCREASE="${EARLY_STOP_INVALID_INCREASE:-0.05}"
EARLY_STOP_HITMAX_INCREASE="${EARLY_STOP_HITMAX_INCREASE:-0.05}"
EARLY_STOP_BEST_ACC_DROP="${EARLY_STOP_BEST_ACC_DROP:-0.0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
EARLY_STOP_QTYPE_ACC_DROP="${EARLY_STOP_QTYPE_ACC_DROP:-0.0}"
EARLY_STOP_QTYPE_DROP_COUNT="${EARLY_STOP_QTYPE_DROP_COUNT:-0}"
PHYSX_RECOVERY="${PHYSX_RECOVERY:-False}"
PHYSX_RECOVERY_METRIC="${PHYSX_RECOVERY_METRIC:-accuracy}"
PHYSX_RECOVERY_MIN_DELTA="${PHYSX_RECOVERY_MIN_DELTA:-0.0}"
PHYSX_RECOVERY_RESTORE_ON_EARLY_STOP="${PHYSX_RECOVERY_RESTORE_ON_EARLY_STOP:-True}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-900}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-24}"
POINTNUM="${POINTNUM:-8192}"
POINT_SCOPE="${POINT_SCOPE:-full_object}"
PROMPT_SUFFIX="${PROMPT_SUFFIX:-}"
QUESTION_TYPE_FILTER="${QUESTION_TYPE_FILTER:-}"
DATASET_PATH="${DATASET_PATH:-$ROOT_DIR/physx_mcq_workspace/PhysX-3D/outputs/physxnet_mcq_verl.json}"
SIDECAR_PATH="${SIDECAR_PATH:-$ROOT_DIR/physx_mcq_workspace/PhysX-3D/outputs/physxnet_mcq.jsonl}"
DATA_DIR_PROVIDED="${DATA_DIR+x}"
DATA_DIR="${DATA_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
RUN_TAG="${RUN_TAG:-}"
MODEL_PATH="${MODEL_PATH:-$(resolve_pointllm_model)}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
DEBUG_STEPS="${DEBUG_STEPS:-}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
TEST_FREQ="${TEST_FREQ:-1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
PRINT_CONFIG="${PRINT_CONFIG:-False}"
LOG_VALIDATION_BATCHES="${LOG_VALIDATION_BATCHES:-False}"
DUMP_VALIDATION_PREDICTIONS="${DUMP_VALIDATION_PREDICTIONS:-True}"
PRINT_VALIDATION_PREDICTIONS="${PRINT_VALIDATION_PREDICTIONS:-False}"
VALIDATION_STEP0_CACHE_JSONL="${VALIDATION_STEP0_CACHE_JSONL:-}"
DUMP_TRAINING_PREDICTIONS="${DUMP_TRAINING_PREDICTIONS:-True}"
PRINT_TRAINING_PREDICTIONS="${PRINT_TRAINING_PREDICTIONS:-False}"
TRAINING_PREDICTIONS_DIR="${TRAINING_PREDICTIONS_DIR:-}"
LOG_TRAIN_GROUPS="${LOG_TRAIN_GROUPS:-}"
LOG_TRAIN_GROUPS_LIMIT="${LOG_TRAIN_GROUPS_LIMIT:-10}"
LOG_TRAIN_RESPONSE_CHARS="${LOG_TRAIN_RESPONSE_CHARS:-160}"
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    RUN_MODE=*) RUN_MODE="${arg#RUN_MODE=}" ;;
    DEBUG=*) DEBUG="${arg#DEBUG=}" ;;
    SUBSET=*) SUBSET="${arg#SUBSET=}" ;;
    VAL_SUBSET=*) VAL_SUBSET="${arg#VAL_SUBSET=}" ;;
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
    VAL_DO_SAMPLE=*) VAL_DO_SAMPLE="${arg#VAL_DO_SAMPLE=}" ;;
    VAL_TEMPERATURE=*) VAL_TEMPERATURE="${arg#VAL_TEMPERATURE=}" ;;
    ACTOR_OPTIMIZER_OFFLOAD=*) ACTOR_OPTIMIZER_OFFLOAD="${arg#ACTOR_OPTIMIZER_OFFLOAD=}" ;;
    ACTOR_FSDP_MP_DTYPE=*) ACTOR_FSDP_MP_DTYPE="${arg#ACTOR_FSDP_MP_DTYPE=}" ;;
    SELECTION=*) SELECTION="${arg#SELECTION=}" ;;
    DATA_SHUFFLE=*) DATA_SHUFFLE="${arg#DATA_SHUFFLE=}" ;;
    ACTOR_LR=*) ACTOR_LR="${arg#ACTOR_LR=}" ;;
    ACTOR_LORA_RANK=*) ACTOR_LORA_RANK="${arg#ACTOR_LORA_RANK=}" ;;
    ACTOR_LORA_ALPHA=*) ACTOR_LORA_ALPHA="${arg#ACTOR_LORA_ALPHA=}" ;;
    ACTOR_LORA_TARGET_MODULES=*) ACTOR_LORA_TARGET_MODULES="${arg#ACTOR_LORA_TARGET_MODULES=}" ;;
    ADV_ESTIMATOR=*) ADV_ESTIMATOR="${arg#ADV_ESTIMATOR=}" ;;
    USE_KL_LOSS=*) USE_KL_LOSS="${arg#USE_KL_LOSS=}" ;;
    KL_LOSS_COEF=*) KL_LOSS_COEF="${arg#KL_LOSS_COEF=}" ;;
    LR_WARMUP_STEPS_RATIO=*) LR_WARMUP_STEPS_RATIO="${arg#LR_WARMUP_STEPS_RATIO=}" ;;
    REWARD_ALPHA=*) REWARD_ALPHA="${arg#REWARD_ALPHA=}" ;;
    FREQUENCY_BETA=*) FREQUENCY_BETA="${arg#FREQUENCY_BETA=}" ;;
    UNKNOWN_REWARD=*) UNKNOWN_REWARD="${arg#UNKNOWN_REWARD=}" ;;
    REWARD_VARIANT=*) REWARD_VARIANT="${arg#REWARD_VARIANT=}" ;;
    TTRL_MIN_MAJORITY_RATIO=*) TTRL_MIN_MAJORITY_RATIO="${arg#TTRL_MIN_MAJORITY_RATIO=}" ;;
    TTRL_MAX_MAJORITY_RATIO=*) TTRL_MAX_MAJORITY_RATIO="${arg#TTRL_MAX_MAJORITY_RATIO=}" ;;
    GEO_NUM_VIEWS=*) GEO_NUM_VIEWS="${arg#GEO_NUM_VIEWS=}" ;;
    GEO_SAMPLES_PER_VIEW=*) GEO_SAMPLES_PER_VIEW="${arg#GEO_SAMPLES_PER_VIEW=}" ;;
    GEO_MIN_VIEW_SUPPORT=*) GEO_MIN_VIEW_SUPPORT="${arg#GEO_MIN_VIEW_SUPPORT=}" ;;
    GEO_MIN_HM=*) GEO_MIN_HM="${arg#GEO_MIN_HM=}" ;;
    GEO_MIN_VIEW_PROB=*) GEO_MIN_VIEW_PROB="${arg#GEO_MIN_VIEW_PROB=}" ;;
    GEO_MIN_SCORE_MARGIN=*) GEO_MIN_SCORE_MARGIN="${arg#GEO_MIN_SCORE_MARGIN=}" ;;
    GEO_SKIP_QUESTION_TYPES=*) GEO_SKIP_QUESTION_TYPES="${arg#GEO_SKIP_QUESTION_TYPES=}" ;;
    GEO_REQUIRE_ORIGINAL_MAJORITY=*) GEO_REQUIRE_ORIGINAL_MAJORITY="${arg#GEO_REQUIRE_ORIGINAL_MAJORITY=}" ;;
    GEO_MIN_ORIGINAL_MAJORITY_RATIO=*) GEO_MIN_ORIGINAL_MAJORITY_RATIO="${arg#GEO_MIN_ORIGINAL_MAJORITY_RATIO=}" ;;
    GEO_SKIP_AMBIGUOUS_JOINT_OPTIONS=*) GEO_SKIP_AMBIGUOUS_JOINT_OPTIONS="${arg#GEO_SKIP_AMBIGUOUS_JOINT_OPTIONS=}" ;;
    GEO_SOFT_GAMMA=*) GEO_SOFT_GAMMA="${arg#GEO_SOFT_GAMMA=}" ;;
    GEO_SOFT_MIN_MAX_PROB=*) GEO_SOFT_MIN_MAX_PROB="${arg#GEO_SOFT_MIN_MAX_PROB=}" ;;
    GEO_SOFT_MIN_KNOWN_COUNT=*) GEO_SOFT_MIN_KNOWN_COUNT="${arg#GEO_SOFT_MIN_KNOWN_COUNT=}" ;;
    POINT_REFRAME_POLICY=*) POINT_REFRAME_POLICY="${arg#POINT_REFRAME_POLICY=}" ;;
    POINT_REFRAME_NUM_VIEWS=*) POINT_REFRAME_NUM_VIEWS="${arg#POINT_REFRAME_NUM_VIEWS=}" ;;
    POINT_REFRAME_SAMPLES_PER_VIEW=*) POINT_REFRAME_SAMPLES_PER_VIEW="${arg#POINT_REFRAME_SAMPLES_PER_VIEW=}" ;;
    POINT_REFRAME_SEED=*) POINT_REFRAME_SEED="${arg#POINT_REFRAME_SEED=}" ;;
    POINT_REFRAME_SCALE_MIN=*) POINT_REFRAME_SCALE_MIN="${arg#POINT_REFRAME_SCALE_MIN=}" ;;
    POINT_REFRAME_SCALE_MAX=*) POINT_REFRAME_SCALE_MAX="${arg#POINT_REFRAME_SCALE_MAX=}" ;;
    POINT_REFRAME_TRANSLATE=*) POINT_REFRAME_TRANSLATE="${arg#POINT_REFRAME_TRANSLATE=}" ;;
    POINT_REFRAME_JITTER_SIGMA=*) POINT_REFRAME_JITTER_SIGMA="${arg#POINT_REFRAME_JITTER_SIGMA=}" ;;
    POINT_REFRAME_JITTER_CLIP=*) POINT_REFRAME_JITTER_CLIP="${arg#POINT_REFRAME_JITTER_CLIP=}" ;;
    POINT_REFRAME_DOWNSAMPLE_MIN=*) POINT_REFRAME_DOWNSAMPLE_MIN="${arg#POINT_REFRAME_DOWNSAMPLE_MIN=}" ;;
    POINT_REFRAME_DOWNSAMPLE_MAX=*) POINT_REFRAME_DOWNSAMPLE_MAX="${arg#POINT_REFRAME_DOWNSAMPLE_MAX=}" ;;
    POINT_REFRAME_RENORMALIZE=*) POINT_REFRAME_RENORMALIZE="${arg#POINT_REFRAME_RENORMALIZE=}" ;;
    PHYSX_EARLY_STOP=*) PHYSX_EARLY_STOP="${arg#PHYSX_EARLY_STOP=}" ;;
    EARLY_STOP_ACC_DROP=*) EARLY_STOP_ACC_DROP="${arg#EARLY_STOP_ACC_DROP=}" ;;
    EARLY_STOP_INVALID_INCREASE=*) EARLY_STOP_INVALID_INCREASE="${arg#EARLY_STOP_INVALID_INCREASE=}" ;;
    EARLY_STOP_HITMAX_INCREASE=*) EARLY_STOP_HITMAX_INCREASE="${arg#EARLY_STOP_HITMAX_INCREASE=}" ;;
    EARLY_STOP_BEST_ACC_DROP=*) EARLY_STOP_BEST_ACC_DROP="${arg#EARLY_STOP_BEST_ACC_DROP=}" ;;
    EARLY_STOP_PATIENCE=*) EARLY_STOP_PATIENCE="${arg#EARLY_STOP_PATIENCE=}" ;;
    EARLY_STOP_MIN_DELTA=*) EARLY_STOP_MIN_DELTA="${arg#EARLY_STOP_MIN_DELTA=}" ;;
    EARLY_STOP_QTYPE_ACC_DROP=*) EARLY_STOP_QTYPE_ACC_DROP="${arg#EARLY_STOP_QTYPE_ACC_DROP=}" ;;
    EARLY_STOP_QTYPE_DROP_COUNT=*) EARLY_STOP_QTYPE_DROP_COUNT="${arg#EARLY_STOP_QTYPE_DROP_COUNT=}" ;;
    PHYSX_RECOVERY=*) PHYSX_RECOVERY="${arg#PHYSX_RECOVERY=}" ;;
    PHYSX_RECOVERY_METRIC=*) PHYSX_RECOVERY_METRIC="${arg#PHYSX_RECOVERY_METRIC=}" ;;
    PHYSX_RECOVERY_MIN_DELTA=*) PHYSX_RECOVERY_MIN_DELTA="${arg#PHYSX_RECOVERY_MIN_DELTA=}" ;;
    PHYSX_RECOVERY_RESTORE_ON_EARLY_STOP=*) PHYSX_RECOVERY_RESTORE_ON_EARLY_STOP="${arg#PHYSX_RECOVERY_RESTORE_ON_EARLY_STOP=}" ;;
    MAX_PROMPT_LENGTH=*) MAX_PROMPT_LENGTH="${arg#MAX_PROMPT_LENGTH=}" ;;
    MAX_RESPONSE_LENGTH=*) MAX_RESPONSE_LENGTH="${arg#MAX_RESPONSE_LENGTH=}" ;;
    POINTNUM=*) POINTNUM="${arg#POINTNUM=}" ;;
    POINT_SCOPE=*) POINT_SCOPE="${arg#POINT_SCOPE=}" ;;
    PROMPT_SUFFIX=*) PROMPT_SUFFIX="${arg#PROMPT_SUFFIX=}" ;;
    QUESTION_TYPE_FILTER=*) QUESTION_TYPE_FILTER="${arg#QUESTION_TYPE_FILTER=}" ;;
    DATASET_PATH=*) DATASET_PATH="${arg#DATASET_PATH=}" ;;
    SIDECAR_PATH=*) SIDECAR_PATH="${arg#SIDECAR_PATH=}" ;;
    DATA_DIR=*) DATA_DIR="${arg#DATA_DIR=}" ;;
    OUTPUT_ROOT=*) OUTPUT_ROOT="${arg#OUTPUT_ROOT=}" ;;
    OUTPUT_DIR=*) OUTPUT_DIR="${arg#OUTPUT_DIR=}" ;;
    RUN_TAG=*) RUN_TAG="${arg#RUN_TAG=}" ;;
    MODEL_PATH=*) MODEL_PATH="${arg#MODEL_PATH=}" ;;
    FORCE_PREPARE=*) FORCE_PREPARE="${arg#FORCE_PREPARE=}" ;;
    DEBUG_STEPS=*) DEBUG_STEPS="${arg#DEBUG_STEPS=}" ;;
    VAL_BEFORE_TRAIN=*) VAL_BEFORE_TRAIN="${arg#VAL_BEFORE_TRAIN=}" ;;
    TEST_FREQ=*) TEST_FREQ="${arg#TEST_FREQ=}" ;;
    SAVE_FREQ=*) SAVE_FREQ="${arg#SAVE_FREQ=}" ;;
    TOTAL_EPOCHS=*) TOTAL_EPOCHS="${arg#TOTAL_EPOCHS=}" ;;
    PRINT_CONFIG=*) PRINT_CONFIG="${arg#PRINT_CONFIG=}" ;;
    LOG_VALIDATION_BATCHES=*) LOG_VALIDATION_BATCHES="${arg#LOG_VALIDATION_BATCHES=}" ;;
    DUMP_VALIDATION_PREDICTIONS=*) DUMP_VALIDATION_PREDICTIONS="${arg#DUMP_VALIDATION_PREDICTIONS=}" ;;
    PRINT_VALIDATION_PREDICTIONS=*) PRINT_VALIDATION_PREDICTIONS="${arg#PRINT_VALIDATION_PREDICTIONS=}" ;;
    VALIDATION_STEP0_CACHE_JSONL=*) VALIDATION_STEP0_CACHE_JSONL="${arg#VALIDATION_STEP0_CACHE_JSONL=}" ;;
    DUMP_TRAINING_PREDICTIONS=*) DUMP_TRAINING_PREDICTIONS="${arg#DUMP_TRAINING_PREDICTIONS=}" ;;
    PRINT_TRAINING_PREDICTIONS=*) PRINT_TRAINING_PREDICTIONS="${arg#PRINT_TRAINING_PREDICTIONS=}" ;;
    TRAINING_PREDICTIONS_DIR=*) TRAINING_PREDICTIONS_DIR="${arg#TRAINING_PREDICTIONS_DIR=}" ;;
    LOG_TRAIN_GROUPS=*) LOG_TRAIN_GROUPS="${arg#LOG_TRAIN_GROUPS=}" ;;
    LOG_TRAIN_GROUPS_LIMIT=*) LOG_TRAIN_GROUPS_LIMIT="${arg#LOG_TRAIN_GROUPS_LIMIT=}" ;;
    LOG_TRAIN_RESPONSE_CHARS=*) LOG_TRAIN_RESPONSE_CHARS="${arg#LOG_TRAIN_RESPONSE_CHARS=}" ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

REWARD_VARIANT_LC="$(echo "$REWARD_VARIANT" | tr '[:upper:]' '[:lower:]')"
if { [ "$REWARD_VARIANT_LC" = "geo_harmony" ] || [ "$REWARD_VARIANT_LC" = "geo_harmony_soft" ]; } && [ "$POINT_REFRAME_POLICY" = "none" ]; then
  POINT_REFRAME_POLICY="sensor_noise"
fi
if [ "$POINT_REFRAME_POLICY" != "none" ] && [ "$POINT_REFRAME_NUM_VIEWS" = "1" ] && [ "$GEO_NUM_VIEWS" != "1" ]; then
  POINT_REFRAME_NUM_VIEWS="$GEO_NUM_VIEWS"
fi
if [ "$POINT_REFRAME_POLICY" != "none" ] && [ "$POINT_REFRAME_SAMPLES_PER_VIEW" = "0" ] && [ "$GEO_SAMPLES_PER_VIEW" != "0" ]; then
  POINT_REFRAME_SAMPLES_PER_VIEW="$GEO_SAMPLES_PER_VIEW"
fi

RUN_MODE=$(echo "$RUN_MODE" | tr '[:upper:]' '[:lower:]')
if [ "$RUN_MODE" != "baseline" ] && [ "$RUN_MODE" != "adapt" ]; then
  echo "RUN_MODE must be baseline or adapt, got: $RUN_MODE" >&2
  exit 1
fi
if [ -z "$ROLLOUT_N" ]; then
  if [ "$DEBUG" = "1" ]; then
    ROLLOUT_N=2
  else
    ROLLOUT_N=16
  fi
fi
if [ -z "$N_SAMPLES_PER_PROMPT" ]; then
  N_SAMPLES_PER_PROMPT="$ROLLOUT_N"
fi
if [ -z "$N_VOTES_PER_PROMPT" ]; then
  N_VOTES_PER_PROMPT="$N_SAMPLES_PER_PROMPT"
fi
if [ -z "$DEBUG_STEPS" ] && [ "$DEBUG" = "1" ] && [ "$RUN_MODE" = "adapt" ]; then
  DEBUG_STEPS=1
fi
if [ -z "$LOG_TRAIN_GROUPS" ]; then
  if [ "$DEBUG" = "1" ]; then
    LOG_TRAIN_GROUPS=True
  else
    LOG_TRAIN_GROUPS=False
  fi
fi
if [ -z "$DATA_DIR" ]; then
  DATA_QTYPE_TAG=""
  if [ -n "$QUESTION_TYPE_FILTER" ]; then
    DATA_QTYPE_TAG="_q$(tag_value "$QUESTION_TYPE_FILTER")"
  fi
  DATA_DIR="$ROOT_DIR/outputs/physx_mcq_verl_train${SUBSET}_val${VAL_SUBSET}_seed${SEED}${DATA_QTYPE_TAG}"
fi
if [ -z "$RUN_TAG" ]; then
  RUN_TAG="$(date +%Y%m%d_%H%M%S)-physx-${RUN_MODE}-train${SUBSET}-val${VAL_SUBSET}-r${ROLLOUT_N}-v${N_VOTES_PER_PROMPT}-a$(tag_value "$REWARD_ALPHA")-b$(tag_value "$FREQUENCY_BETA")"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_TAG}"
VALIDATION_PREDICTIONS_DIR="${VALIDATION_PREDICTIONS_DIR:-$OUTPUT_DIR}"
TRAINING_PREDICTIONS_DIR="${TRAINING_PREDICTIONS_DIR:-$OUTPUT_DIR}"

if [ "$NO_GPU" -gt 1 ]; then
  echo "This script defaults to one GPU per AGENTS.md. You explicitly set NO_GPU=$NO_GPU." >&2
fi
if [ ! -d "$POINTLLM_DIR" ]; then
  echo "Missing local PointLLM source at $POINTLLM_DIR; refusing to download." >&2
  exit 1
fi
python -c "import pointllm" >/dev/null

mkdir -p "$OUTPUT_DIR"
if [ "$FORCE_PREPARE" = "1" ] || [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
  python "$POINTLLM_DIR/scripts/prepare_physx_mcq_verl.py" \
    --dataset-path "$DATASET_PATH" \
    --sidecar-path "$SIDECAR_PATH" \
    --output-dir "$DATA_DIR" \
    --train-subset "$SUBSET" \
    --val-subset "$VAL_SUBSET" \
    --seed "$SEED" \
    --pointnum "$POINTNUM" \
    --point-scope "$POINT_SCOPE" \
    --prompt-suffix "$PROMPT_SUFFIX" \
    --selection "$SELECTION" \
    --question-type-filter "$QUESTION_TYPE_FILTER" \
    --overwrite
fi

RUN_ARGS=(
  algorithm.adv_estimator="$ADV_ESTIMATOR"
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
  data.train_files=["$DATA_DIR/train.parquet"]
  data.val_files=["$DATA_DIR/test.parquet"]
  data.cache_dir=/root/autodl-tmp/.cache/verl/rlhf
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.val_batch_size="$VAL_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.point_cloud_num_points="$POINTNUM"
  data.shuffle="$DATA_SHUFFLE"
  data.filter_overlong_prompts=False
  data.truncation=error
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.external_lib=pointllm.model
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.model.lora_rank="$ACTOR_LORA_RANK"
  actor_rollout_ref.model.lora_alpha="$ACTOR_LORA_ALPHA"
  actor_rollout_ref.model.target_modules="$ACTOR_LORA_TARGET_MODULES"
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS"
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO"
  actor_rollout_ref.actor.optim.warmup_style=cosine
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD"
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
  +actor_rollout_ref.rollout.point_cloud_reframe_policy="$POINT_REFRAME_POLICY"
  +actor_rollout_ref.rollout.point_cloud_reframe_num_views="$POINT_REFRAME_NUM_VIEWS"
  +actor_rollout_ref.rollout.point_cloud_reframe_samples_per_view="$POINT_REFRAME_SAMPLES_PER_VIEW"
  +actor_rollout_ref.rollout.point_cloud_reframe_seed="$POINT_REFRAME_SEED"
  +actor_rollout_ref.rollout.point_cloud_reframe_scale_min="$POINT_REFRAME_SCALE_MIN"
  +actor_rollout_ref.rollout.point_cloud_reframe_scale_max="$POINT_REFRAME_SCALE_MAX"
  +actor_rollout_ref.rollout.point_cloud_reframe_translate="$POINT_REFRAME_TRANSLATE"
  +actor_rollout_ref.rollout.point_cloud_reframe_jitter_sigma="$POINT_REFRAME_JITTER_SIGMA"
  +actor_rollout_ref.rollout.point_cloud_reframe_jitter_clip="$POINT_REFRAME_JITTER_CLIP"
  +actor_rollout_ref.rollout.point_cloud_reframe_downsample_min="$POINT_REFRAME_DOWNSAMPLE_MIN"
  +actor_rollout_ref.rollout.point_cloud_reframe_downsample_max="$POINT_REFRAME_DOWNSAMPLE_MAX"
  +actor_rollout_ref.rollout.point_cloud_reframe_renormalize="$POINT_REFRAME_RENORMALIZE"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE"
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_ROLLOUT_N"
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE"
  actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P"
  actor_rollout_ref.rollout.val_kwargs.top_k=0
  reward_model.reward_manager=batch
  +reward_model.val_num_examine=4
  reward_model.reward_kwargs.alpha="$REWARD_ALPHA"
  +reward_model.reward_kwargs.frequency_beta="$FREQUENCY_BETA"
  +reward_model.reward_kwargs.unknown_reward="$UNKNOWN_REWARD"
  +reward_model.reward_kwargs.reward_variant="$REWARD_VARIANT"
  +reward_model.reward_kwargs.ttrl_min_majority_ratio="$TTRL_MIN_MAJORITY_RATIO"
  +reward_model.reward_kwargs.ttrl_max_majority_ratio="$TTRL_MAX_MAJORITY_RATIO"
  +reward_model.reward_kwargs.geo_num_views="$GEO_NUM_VIEWS"
  +reward_model.reward_kwargs.geo_samples_per_view="$GEO_SAMPLES_PER_VIEW"
  +reward_model.reward_kwargs.geo_min_view_support="$GEO_MIN_VIEW_SUPPORT"
  +reward_model.reward_kwargs.geo_min_hm="$GEO_MIN_HM"
  +reward_model.reward_kwargs.geo_min_view_prob="$GEO_MIN_VIEW_PROB"
  +reward_model.reward_kwargs.geo_min_score_margin="$GEO_MIN_SCORE_MARGIN"
  +reward_model.reward_kwargs.geo_skip_question_types="$GEO_SKIP_QUESTION_TYPES"
  +reward_model.reward_kwargs.geo_require_original_majority="$GEO_REQUIRE_ORIGINAL_MAJORITY"
  +reward_model.reward_kwargs.geo_min_original_majority_ratio="$GEO_MIN_ORIGINAL_MAJORITY_RATIO"
  +reward_model.reward_kwargs.geo_skip_ambiguous_joint_options="$GEO_SKIP_AMBIGUOUS_JOINT_OPTIONS"
  +reward_model.reward_kwargs.geo_soft_gamma="$GEO_SOFT_GAMMA"
  +reward_model.reward_kwargs.geo_soft_min_max_prob="$GEO_SOFT_MIN_MAX_PROB"
  +reward_model.reward_kwargs.geo_soft_min_known_count="$GEO_SOFT_MIN_KNOWN_COUNT"
  +reward_model.reward_kwargs.train_group_size="$N_VOTES_PER_PROMPT"
  +reward_model.reward_kwargs.log_train_groups="$LOG_TRAIN_GROUPS"
  +reward_model.reward_kwargs.log_train_groups_limit="$LOG_TRAIN_GROUPS_LIMIT"
  +reward_model.reward_kwargs.log_train_response_chars="$LOG_TRAIN_RESPONSE_CHARS"
  reward_model.reward_kwargs.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT"
  reward_model.reward_kwargs.n_votes_per_prompt="$N_VOTES_PER_PROMPT"
  custom_reward_function.path="$VERL_DIR/verl/utils/reward_score/physx_mcq_ttrv.py"
  custom_reward_function.name=compute_score
  trainer.logger=['console']
  +trainer.print_config="$PRINT_CONFIG"
  +trainer.log_validation_batches="$LOG_VALIDATION_BATCHES"
  +trainer.dump_validation_predictions="$DUMP_VALIDATION_PREDICTIONS"
  +trainer.print_validation_predictions="$PRINT_VALIDATION_PREDICTIONS"
  +trainer.validation_predictions_dir="$VALIDATION_PREDICTIONS_DIR"
  +trainer.validation_step0_cache_jsonl="$VALIDATION_STEP0_CACHE_JSONL"
  +trainer.dump_training_predictions="$DUMP_TRAINING_PREDICTIONS"
  +trainer.print_training_predictions="$PRINT_TRAINING_PREDICTIONS"
  +trainer.training_predictions_dir="$TRAINING_PREDICTIONS_DIR"
  +trainer.physx_early_stop_enabled="$PHYSX_EARLY_STOP"
  +trainer.physx_early_stop_acc_drop="$EARLY_STOP_ACC_DROP"
  +trainer.physx_early_stop_invalid_increase="$EARLY_STOP_INVALID_INCREASE"
  +trainer.physx_early_stop_hitmax_increase="$EARLY_STOP_HITMAX_INCREASE"
  +trainer.physx_early_stop_best_acc_drop="$EARLY_STOP_BEST_ACC_DROP"
  +trainer.physx_early_stop_patience="$EARLY_STOP_PATIENCE"
  +trainer.physx_early_stop_min_delta="$EARLY_STOP_MIN_DELTA"
  +trainer.physx_early_stop_qtype_acc_drop="$EARLY_STOP_QTYPE_ACC_DROP"
  +trainer.physx_early_stop_qtype_drop_count="$EARLY_STOP_QTYPE_DROP_COUNT"
  +trainer.physx_recovery_enabled="$PHYSX_RECOVERY"
  +trainer.physx_recovery_metric="$PHYSX_RECOVERY_METRIC"
  +trainer.physx_recovery_min_delta="$PHYSX_RECOVERY_MIN_DELTA"
  +trainer.physx_recovery_restore_on_early_stop="$PHYSX_RECOVERY_RESTORE_ON_EARLY_STOP"
  trainer.project_name=ttrv-3d
  trainer.experiment_name=pointllm-physx-mcq-"$RUN_MODE"-train"$SUBSET"-val"$VAL_SUBSET"
  trainer.n_gpus_per_node="$NO_GPU"
  trainer.nnodes=1
  trainer.val_before_train="$VAL_BEFORE_TRAIN"
  trainer.test_freq="$TEST_FREQ"
  trainer.save_freq="$SAVE_FREQ"
  trainer.total_epochs="$TOTAL_EPOCHS"
  trainer.default_local_dir="$OUTPUT_DIR/checkpoints"
)

if [ -n "$ACTOR_FSDP_MP_DTYPE" ]; then
  RUN_ARGS+=(
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype="$ACTOR_FSDP_MP_DTYPE"
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype="$ACTOR_FSDP_MP_DTYPE"
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype="$ACTOR_FSDP_MP_DTYPE"
  )
fi

if [ "$RUN_MODE" = "baseline" ]; then
  RUN_ARGS+=(+trainer.val_only=True)
fi
if [ -n "$DEBUG_STEPS" ]; then
  RUN_ARGS+=(trainer.total_training_steps="$DEBUG_STEPS")
fi

COMMAND_FILE="$OUTPUT_DIR/command.sh"
RUN_CONFIG_FILE="$OUTPUT_DIR/run_config.txt"
LOG_FILE="$OUTPUT_DIR/run.log"
{
  printf '#!/bin/bash\n'
  printf 'set -euo pipefail\n'
  printf 'cd %q\n' "$ROOT_DIR"
  printf 'bash %q' "$SCRIPT_DIR/run_physx_mcq_pointllm.sh"
  for arg in "${ORIGINAL_ARGS[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
} > "$COMMAND_FILE"
chmod +x "$COMMAND_FILE"
{
  printf 'run_mode=%s\n' "$RUN_MODE"
  printf 'run_tag=%s\n' "$RUN_TAG"
  printf 'output_dir=%s\n' "$OUTPUT_DIR"
  printf 'data_dir=%s\n' "$DATA_DIR"
  printf 'model_path=%s\n' "$MODEL_PATH"
  printf 'no_gpu=%s\n' "$NO_GPU"
  printf 'data_shuffle=%s\n' "$DATA_SHUFFLE"
  printf 'train_batch_size=%s\n' "$TRAIN_BATCH_SIZE"
  printf 'val_batch_size=%s\n' "$VAL_BATCH_SIZE"
  printf 'adv_estimator=%s\n' "$ADV_ESTIMATOR"
  printf 'actor_lora_rank=%s\n' "$ACTOR_LORA_RANK"
  printf 'actor_lora_alpha=%s\n' "$ACTOR_LORA_ALPHA"
  printf 'actor_lora_target_modules=%s\n' "$ACTOR_LORA_TARGET_MODULES"
  printf 'rollout_n=%s\n' "$ROLLOUT_N"
  printf 'n_samples_per_prompt=%s\n' "$N_SAMPLES_PER_PROMPT"
  printf 'n_votes_per_prompt=%s\n' "$N_VOTES_PER_PROMPT"
  printf 'reward_alpha=%s\n' "$REWARD_ALPHA"
  printf 'frequency_beta=%s\n' "$FREQUENCY_BETA"
  printf 'unknown_reward=%s\n' "$UNKNOWN_REWARD"
  printf 'reward_variant=%s\n' "$REWARD_VARIANT"
  printf 'ttrl_min_majority_ratio=%s\n' "$TTRL_MIN_MAJORITY_RATIO"
  printf 'ttrl_max_majority_ratio=%s\n' "$TTRL_MAX_MAJORITY_RATIO"
  printf 'geo_num_views=%s\n' "$GEO_NUM_VIEWS"
  printf 'geo_samples_per_view=%s\n' "$GEO_SAMPLES_PER_VIEW"
  printf 'geo_min_view_support=%s\n' "$GEO_MIN_VIEW_SUPPORT"
  printf 'geo_min_hm=%s\n' "$GEO_MIN_HM"
  printf 'geo_min_view_prob=%s\n' "$GEO_MIN_VIEW_PROB"
  printf 'geo_min_score_margin=%s\n' "$GEO_MIN_SCORE_MARGIN"
  printf 'geo_skip_question_types=%s\n' "$GEO_SKIP_QUESTION_TYPES"
  printf 'geo_require_original_majority=%s\n' "$GEO_REQUIRE_ORIGINAL_MAJORITY"
  printf 'geo_min_original_majority_ratio=%s\n' "$GEO_MIN_ORIGINAL_MAJORITY_RATIO"
  printf 'geo_skip_ambiguous_joint_options=%s\n' "$GEO_SKIP_AMBIGUOUS_JOINT_OPTIONS"
  printf 'geo_soft_gamma=%s\n' "$GEO_SOFT_GAMMA"
  printf 'geo_soft_min_max_prob=%s\n' "$GEO_SOFT_MIN_MAX_PROB"
  printf 'geo_soft_min_known_count=%s\n' "$GEO_SOFT_MIN_KNOWN_COUNT"
  printf 'point_reframe_policy=%s\n' "$POINT_REFRAME_POLICY"
  printf 'point_reframe_num_views=%s\n' "$POINT_REFRAME_NUM_VIEWS"
  printf 'point_reframe_samples_per_view=%s\n' "$POINT_REFRAME_SAMPLES_PER_VIEW"
  printf 'point_reframe_seed=%s\n' "$POINT_REFRAME_SEED"
  printf 'point_reframe_scale_min=%s\n' "$POINT_REFRAME_SCALE_MIN"
  printf 'point_reframe_scale_max=%s\n' "$POINT_REFRAME_SCALE_MAX"
  printf 'point_reframe_translate=%s\n' "$POINT_REFRAME_TRANSLATE"
  printf 'point_reframe_jitter_sigma=%s\n' "$POINT_REFRAME_JITTER_SIGMA"
  printf 'point_reframe_jitter_clip=%s\n' "$POINT_REFRAME_JITTER_CLIP"
  printf 'point_reframe_downsample_min=%s\n' "$POINT_REFRAME_DOWNSAMPLE_MIN"
  printf 'point_reframe_downsample_max=%s\n' "$POINT_REFRAME_DOWNSAMPLE_MAX"
  printf 'point_reframe_renormalize=%s\n' "$POINT_REFRAME_RENORMALIZE"
  printf 'physx_early_stop=%s\n' "$PHYSX_EARLY_STOP"
  printf 'validation_step0_cache_jsonl=%s\n' "$VALIDATION_STEP0_CACHE_JSONL"
  printf 'early_stop_acc_drop=%s\n' "$EARLY_STOP_ACC_DROP"
  printf 'early_stop_invalid_increase=%s\n' "$EARLY_STOP_INVALID_INCREASE"
  printf 'early_stop_hitmax_increase=%s\n' "$EARLY_STOP_HITMAX_INCREASE"
  printf 'early_stop_best_acc_drop=%s\n' "$EARLY_STOP_BEST_ACC_DROP"
  printf 'early_stop_patience=%s\n' "$EARLY_STOP_PATIENCE"
  printf 'early_stop_min_delta=%s\n' "$EARLY_STOP_MIN_DELTA"
  printf 'early_stop_qtype_acc_drop=%s\n' "$EARLY_STOP_QTYPE_ACC_DROP"
  printf 'early_stop_qtype_drop_count=%s\n' "$EARLY_STOP_QTYPE_DROP_COUNT"
  printf 'physx_recovery=%s\n' "$PHYSX_RECOVERY"
  printf 'physx_recovery_metric=%s\n' "$PHYSX_RECOVERY_METRIC"
  printf 'physx_recovery_min_delta=%s\n' "$PHYSX_RECOVERY_MIN_DELTA"
  printf 'physx_recovery_restore_on_early_stop=%s\n' "$PHYSX_RECOVERY_RESTORE_ON_EARLY_STOP"
  printf 'point_scope=%s\n' "$POINT_SCOPE"
  printf 'pointnum=%s\n' "$POINTNUM"
  printf 'prompt_suffix=%s\n' "$PROMPT_SUFFIX"
  printf 'question_type_filter=%s\n' "$QUESTION_TYPE_FILTER"
  printf 'dataset_path=%s\n' "$DATASET_PATH"
  printf 'sidecar_path=%s\n' "$SIDECAR_PATH"
  printf 'dump_training_predictions=%s\n' "$DUMP_TRAINING_PREDICTIONS"
  printf 'training_predictions_dir=%s\n' "$TRAINING_PREDICTIONS_DIR"
  printf 'primary_eval_metric=%s\n' "parsed_abcd_accuracy"
  printf 'ttrv_frequency_space=%s\n' "parsed_prediction_A_B_C_D_unknown"
} > "$RUN_CONFIG_FILE"

set +e
python -m verl.trainer.main_ppo "${RUN_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [ "$RUN_STATUS" -ne 0 ] && [ ! -f "$OUTPUT_DIR/early_stop_status.json" ]; then
  python - "$OUTPUT_DIR" "$RUN_STATUS" "$LOG_FILE" "$EARLY_STOP_ACC_DROP" "$EARLY_STOP_INVALID_INCREASE" "$EARLY_STOP_HITMAX_INCREASE" <<'PY_FAILURE' || true
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
status = int(sys.argv[2])
log_path = Path(sys.argv[3])
acc_drop = float(sys.argv[4])
invalid_increase = float(sys.argv[5])
hitmax_increase = float(sys.argv[6])
reason = "runtime_failure"
try:
    log_text = log_path.read_text(errors="ignore").lower()
    if "out of memory" in log_text or "cuda oom" in log_text or "cuda error: out of memory" in log_text:
        reason = "oom"
except Exception:
    pass
payload = {
    "enabled": True,
    "triggered": True,
    "reason": reason,
    "step": None,
    "run_status": status,
    "baseline": None,
    "current": None,
    "thresholds": {
        "acc_drop": acc_drop,
        "invalid_increase": invalid_increase,
        "hitmax_increase": hitmax_increase,
    },
}
(out / "early_stop_status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
(out / "early_stop_reason.txt").write_text(reason + "\n")
PY_FAILURE
fi

summarize_prediction_files() {
  local prefix="$1"
  shift || true
  local pred base step
  for pred in "$@"; do
    [ -f "$pred" ] || continue
    base=$(basename "$pred" .jsonl)
    step="${base##*_}"
    python "$VERL_DIR/verl/utils/reward_score/physx_mcq_ttrv.py" \
      --predictions-jsonl "$pred" \
      --metrics-json "$OUTPUT_DIR/${prefix}_metrics_${step}.json" \
      --report-md "$OUTPUT_DIR/${prefix}_report_${step}.md" \
      --title "PointLLM PhysX-MCQ ${RUN_MODE} ${prefix} ${step} train=${SUBSET} val=${VAL_SUBSET}" || true
  done
}

shopt -s nullglob
TRAIN_PRED_FILES=("$TRAINING_PREDICTIONS_DIR"/training_predictions_step*.jsonl)
VAL_PRED_FILES=("$VALIDATION_PREDICTIONS_DIR"/validation_predictions_step*.jsonl)
shopt -u nullglob
summarize_prediction_files train "${TRAIN_PRED_FILES[@]}"
summarize_prediction_files val "${VAL_PRED_FILES[@]}"

LATEST_PRED=$(ls -1 "$VALIDATION_PREDICTIONS_DIR"/validation_predictions_step*.jsonl 2>/dev/null | sort | tail -1 || true)
if [ -n "$LATEST_PRED" ]; then
  python "$VERL_DIR/verl/utils/reward_score/physx_mcq_ttrv.py" \
    --predictions-jsonl "$LATEST_PRED" \
    --metrics-json "$OUTPUT_DIR/metrics.json" \
    --report-md "$OUTPUT_DIR/report.md" \
    --title "PointLLM PhysX-MCQ ${RUN_MODE} train=${SUBSET} val=${VAL_SUBSET}" || true
fi

python - "$OUTPUT_DIR" <<'PY_SUMMARY' || true
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
metrics_paths = sorted(out.glob("val_metrics_step*.json"))
rows = []
for path in metrics_paths:
    try:
        metrics = json.loads(path.read_text())
    except Exception:
        continue
    match = re.search(r"step(\d+)", path.stem)
    step = int(match.group(1)) if match else None
    ttrv = metrics.get("ttrv_metrics", {}) or {}
    rows.append({
        "step": step,
        "metrics_path": str(path),
        "predictions_jsonl": metrics.get("predictions_jsonl"),
        "accuracy": metrics.get("accuracy"),
        "num_evaluated": metrics.get("num_evaluated"),
        "invalid_outputs": metrics.get("invalid_outputs"),
        "majority_accuracy": ttrv.get("majority_accuracy"),
        "pass_at_votes": ttrv.get("pass_at_votes"),
        "best_accuracy_at_votes": ttrv.get("best_accuracy_at_votes"),
        "worst_accuracy_at_votes": ttrv.get("worst_accuracy_at_votes"),
        "frequency_mean": ttrv.get("frequency_mean"),
        "normalized_entropy_mean": ttrv.get("normalized_entropy_mean"),
        "ttrv_reward_mean": ttrv.get("ttrv_reward_mean"),
    })
if rows:
    best = max(rows, key=lambda row: (-1.0 if row["accuracy"] is None else float(row["accuracy"]), -(row["step"] or 0)))
    payload = {"selection_metric": "accuracy", "best_step": best["step"], "best": best, "steps": rows}
    (out / "best_validation_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Best Validation Step",
        "",
        "Selection metric: `accuracy` over parsed A/B/C/D/unknown predictions.",
        "",
        "| step | acc | majority_acc | pass@votes | best_acc@votes | worst_acc@votes | invalid | freq_mean | norm_entropy | ttrv_reward |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    def fmt(value):
        return "n/a" if value is None else f"{float(value):.6f}"
    for row in rows:
        invalid = row["invalid_outputs"] if row["invalid_outputs"] is not None else "n/a"
        lines.append(
            f"| {row['step']} | {fmt(row['accuracy'])} | {fmt(row['majority_accuracy'])} | {fmt(row['pass_at_votes'])} | {fmt(row['best_accuracy_at_votes'])} | {fmt(row['worst_accuracy_at_votes'])} | {invalid} | {fmt(row['frequency_mean'])} | {fmt(row['normalized_entropy_mean'])} | {fmt(row['ttrv_reward_mean'])} |"
        )
    lines.extend(["", f"Best step: `{best['step']}`", f"Best metrics: `{best['metrics_path']}`"])
    (out / "best_validation_report.md").write_text("\n".join(lines) + "\n")
PY_SUMMARY

python - "$OUTPUT_DIR" "$RUN_STATUS" <<'PY_ANALYSIS' || true
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
run_status = int(sys.argv[2])

def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def step_from_path(path):
    match = re.search(r"step(\d+)", path.stem)
    return int(match.group(1)) if match else None

def fmt(value):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)

def metric_rows(prefix):
    rows = []
    for path in sorted(out.glob(f"{prefix}_metrics_step*.json")):
        metrics = load_json(path)
        if not metrics:
            continue
        rows.append((step_from_path(path), path, metrics))
    return rows

val_rows = metric_rows("val")
train_rows = metric_rows("train")
early = load_json(out / "early_stop_status.json") or {"triggered": False, "reason": "not_configured"}
recovery = load_json(out / "physx_recovery_status.json")
best = load_json(out / "best_validation_metrics.json") or {}
best_step = best.get("best_step")

lines = [
    "# PhysX-MCQ TTRL Majority Analysis",
    "",
    f"- Run status: `{run_status}`",
    f"- Early stop: `{early.get('reason', 'unknown')}`",
    f"- Weight recovery: `{('restored' if recovery and recovery.get('restored') else 'tracked' if recovery else 'not_configured')}`",
    f"- Full-run gate: `{'blocked' if early.get('triggered') or run_status != 0 else 'passed'}`",
    f"- Best validation step: `{best_step if best_step is not None else 'n/a'}`",
    "",
    "## Validation Steps",
    "",
    "| step | acc | invalid | hit_max | resp_len_mean | joint_type | movable_part | object_category |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for step, _path, metrics in val_rows:
    by_type = metrics.get("accuracy_by_question_type", {})
    lines.append(
        "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            step,
            fmt(metrics.get("accuracy")),
            metrics.get("invalid_outputs", "n/a"),
            fmt(metrics.get("hit_max_response_length_rate")),
            fmt(metrics.get("response_token_len_mean")),
            fmt((by_type.get("joint_type") or {}).get("accuracy")),
            fmt((by_type.get("movable_part") or {}).get("accuracy")),
            fmt((by_type.get("object_category") or {}).get("accuracy")),
        )
    )

lines.extend([
    "",
    "## Training Pseudo Labels",
    "",
    "| step | response_acc | majority_acc | pass@votes | invalid_rate | ttrl_reward_mean | majority_ratio_mean | tie_rate | geo_hm | geo_support | geo_skip | soft_selected | soft_top_prob | soft_gt_mass |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
])
for step, _path, metrics in train_rows:
    ttrv = metrics.get("ttrv_metrics", {})
    lines.append(
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            step,
            fmt(ttrv.get("response_accuracy")),
            fmt(ttrv.get("majority_accuracy")),
            fmt(ttrv.get("pass_at_votes")),
            fmt(ttrv.get("vote_invalid_rate")),
            fmt(ttrv.get("ttrv_reward_mean")),
            fmt(ttrv.get("ttrl_majority_ratio_mean")),
            fmt(ttrv.get("ttrl_majority_tie_mean")),
            fmt(ttrv.get("geo_harmony_hm_mean")),
            fmt(ttrv.get("geo_view_support_mean")),
            fmt(ttrv.get("geo_skipped_rate")),
            fmt(ttrv.get("geo_soft_selected_rate")),
            fmt(ttrv.get("geo_soft_top_prob_mean")),
            fmt(ttrv.get("geo_soft_gt_mass_mean")),
        )
    )

latest_val = val_rows[-1][2] if val_rows else {}
lines.extend([
    "",
    "## Latest Prediction Distribution",
    "",
    "| prediction | count |",
    "| --- | ---: |",
])
for pred, count in (latest_val.get("prediction_distribution") or {}).items():
    lines.append(f"| {pred} | {count} |")

if early:
    lines.extend([
        "",
        "## Early Stop Details",
        "",
        "```json",
        json.dumps(early, indent=2, ensure_ascii=False),
        "```",
    ])

if recovery:
    lines.extend([
        "",
        "## Weight Recovery Details",
        "",
        "```json",
        json.dumps(recovery, indent=2, ensure_ascii=False),
        "```",
    ])

(out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY_ANALYSIS

exit "$RUN_STATUS"
