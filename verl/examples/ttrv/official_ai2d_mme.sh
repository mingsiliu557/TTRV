#!/usr/bin/env bash
# This entry point only delegates to the existing trainer; no alternate model stack.
set -Eeuo pipefail
: "${DATA_LOCAL_DIR:?Set the prepared official-data root}"
: "${BACKBONE_PATH:?Set a local InternVL3-2B snapshot}"
: "${OFFICIAL_RUN_DIR:?Set a fresh absolute run directory}"
: "${TASK:?AI2D_TEST or MME}"
: "${METHOD:?base, ttrv or prove}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ "$OFFICIAL_RUN_DIR" = /* && ! -e "$OFFICIAL_RUN_DIR" ]] || { echo 'Run directory must be fresh and absolute' >&2; exit 2; }
mkdir -p "$OFFICIAL_RUN_DIR"
export TTRL_EVAL_OUTPUT_JSONL="$OFFICIAL_RUN_DIR/predictions.raw.jsonl"
export TTRL_TRAIN_ROLLOUT_JSONL="$OFFICIAL_RUN_DIR/train_rollouts.jsonl"
export LOG_FILE="$OFFICIAL_RUN_DIR/train.log" LOG_DIR="$OFFICIAL_RUN_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_COT=0
unset TTRL_SELECTION_EVAL TTRL_EVAL_GROUP_OUTPUT_JSONL
steps=${OFFICIAL_STEPS:-20}
val_file="$DATA_LOCAL_DIR/$TASK/test.parquet"
if [[ "${OFFICIAL_SMOKE:-0}" == 1 ]]; then val_file="$DATA_LOCAL_DIR/$TASK/smoke.parquet"; fi
base_args=(trainer.val_before_train=False)
if [[ "$METHOD" == base ]]; then base_args=(trainer.val_before_train=True +trainer.val_only=True); fi
exec bash "$SCRIPT_DIR/run.sh" \
  data.filter_overlong_prompts=False data.truncation=error \
  +data.dataloader_num_workers=0 +data.validation_batch_size=2 +data.seed=42 \
  "data.val_files=[$val_file]" \
  +runtime.low_host_memory=True +runtime.ray_num_cpus=4 +runtime.object_store_memory=134217728 \
  +actor_rollout_ref.model.keep_reference_on_gpu=True \
  +actor_rollout_ref.seed=42 +actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  +actor_rollout_ref.rollout.enable_sleep_mode=False +actor_rollout_ref.rollout.swap_space=0 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.actor.use_torch_compile=False \
  trainer.resume_mode=disable trainer.save_freq=-1 \
  "trainer.total_training_steps=$steps" "trainer.test_freq=$steps" \
  "trainer.default_local_dir=$OFFICIAL_RUN_DIR/checkpoints" \
  "${base_args[@]}" "$@"
