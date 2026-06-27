#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

RESOURCE_QUEUE_ID="${RESOURCE_QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"
DATA_ROOT="${DATA_ROOT:-/jiigan-hp/ttrv-datasets}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/vepfs_default/chanxueyan/lhp/lms/ttrv_runtime}"
BACKBONE_PATH="${BACKBONE_PATH:-$RUNTIME_ROOT/models/OpenGVLab/InternVL3-2B}"
PYTHON_BIN="${PYTHON_BIN:-/vepfs_default/chanxueyan/lhp/lms/envs/ttrv/bin/python}"
RUN_TAG="${RUN_TAG:-direct_answer_ttrv_$(date +%Y%m%d_%H%M%S)}"
DATASETS=(${DATASETS:-visualsimpleqa ocrbench_v1 aokvqa_da_val})

launch_worker() {
  local name="$1"
  local inner_cmd="$2"
  echo "[submit] $name"
  volc ml_devinstance launch     --resource_queue_id "$RESOURCE_QUEUE_ID"     --flavor_id "$FLAVOR_ID"     zsh -lc "$inner_cmd"
}

final_task_for_key() {
  case "$1" in
    visualsimpleqa) echo "visualsimpleqa_da_20_hq_major_correct" ;;
    ocrbench_v1) echo "ocrbench_v1_20_hq_major_correct" ;;
    aokvqa_da_val) echo "aokvqa_da_val_20_hq_major_correct" ;;
    *) echo "unknown dataset key: $1" >&2; return 2 ;;
  esac
}

for key in "${DATASETS[@]}"; do
  final_task=$(final_task_for_key "$key")
  final_dir="$DATA_ROOT/verl_data/$final_task"
  if [[ ! -s "$final_dir/train.parquet" || ! -s "$final_dir/test.parquet" ]]; then
    echo "[error] missing HQ20 parquet for $key: $final_dir" >&2
    exit 1
  fi

  launch_worker "ttrv_${key}" "
    set -Eeuo pipefail
    cd '$ROOT_DIR'
    export DATA_ROOT='$DATA_ROOT'
    export RUNTIME_ROOT='$RUNTIME_ROOT'
    export BACKBONE_PATH='$BACKBONE_PATH'
    export PYTHON_BIN='$PYTHON_BIN'
    export CUDA_VISIBLE_DEVICES=0,1
    export NO_GPU=2
    export RUN_KIND=method
    export RUN_TAG='${RUN_TAG}_${key}_frequency_entropy'
    export DATASETS='$key'
    export TTRL_REWARD_STYLE=frequency_entropy
    export ENTROPY_COEF=0.75
    export USE_PAPO_PRCP_LOSS=false
    export DENSITY_EMBEDDING_SCOPE=response_mean_pool
    bash run_direct_answer_vqa_experiments.sh
  "
done

echo "[submit] complete tag=$RUN_TAG"
