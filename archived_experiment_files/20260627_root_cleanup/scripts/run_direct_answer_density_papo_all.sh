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
RUN_TAG="${RUN_TAG:-direct_answer_density_papo_$(date +%Y%m%d_%H%M%S)}"
DATASETS=(${DATASETS:-visualsimpleqa ocrbench_v1 aokvqa_da_val})

launch_worker() {
  local name="$1"
  local inner_cmd="$2"
  echo "[submit] $name"
  volc ml_devinstance launch \
    --resource_queue_id "$RESOURCE_QUEUE_ID" \
    --flavor_id "$FLAVOR_ID" \
    zsh -lc "$inner_cmd"
}

task_names() {
  local key="$1"
  case "$key" in
    visualsimpleqa)
      echo "visualsimpleqa_da_hq_pool80_seed42 visualsimpleqa_da visualsimpleqa_da_20_hq_major_correct"
      ;;
    ocrbench_v1)
      echo "ocrbench_v1_hq_pool80_seed42 ocrbench_v1 ocrbench_v1_20_hq_major_correct"
      ;;
    aokvqa_da_val)
      echo "aokvqa_da_val_hq_pool80_seed42 aokvqa_da_val aokvqa_da_val_20_hq_major_correct"
      ;;
    *)
      echo "unknown dataset key: $key" >&2
      return 2
      ;;
  esac
}

for key in "${DATASETS[@]}"; do
  read -r pool_task source_task final_task < <(task_names "$key")
  final_dir="$DATA_ROOT/verl_data/$final_task"
  if [[ -s "$final_dir/train.parquet" && -s "$final_dir/test.parquet" ]]; then
    echo "[hq-scan] skip existing $final_task"
  else
    launch_worker "hq_scan_${key}" "
      set -Eeuo pipefail
      cd '$ROOT_DIR'
      export BACKBONE_PATH='$BACKBONE_PATH'
      export PYTHON_BIN='$PYTHON_BIN'
      export DATA_LOCAL_DIR='$DATA_ROOT/verl_data'
      export HF_HOME='$DATA_ROOT/hf_home'
      export RUNTIME_ROOT='$RUNTIME_ROOT'
      export RUN_TAG='${RUN_TAG}_hq_${key}'
      TASK='$pool_task' SOURCE_TASK='$source_task' FINAL_TASK='$final_task' \
      ANSWER_PARSE_MODE=direct_answer_short ANSWER_CHOICE_LABELS=A-D HQ_METRIC=direct_answer HQ_SELECTION_MODE=best_of_n HQ_REPEAT_FILL=1 \
      bash run_hq_majority_scan.sh
    "
  fi

  launch_worker "density_papo_${key}" "
    set -Eeuo pipefail
    cd '$ROOT_DIR'
    export DATA_ROOT='$DATA_ROOT'
    export RUNTIME_ROOT='$RUNTIME_ROOT'
    export BACKBONE_PATH='$BACKBONE_PATH'
    export PYTHON_BIN='$PYTHON_BIN'
    export CUDA_VISIBLE_DEVICES=0,1
    export NO_GPU=2
    export RUN_KIND=method
    export RUN_TAG='${RUN_TAG}_${key}_density_cluster_answer_entropy'
    export DATASETS='$key'
    export TTRL_REWARD_STYLE=density_cluster_answer_entropy
    export USE_PAPO_PRCP_LOSS=true
    export DENSITY_TEMPERATURE_T0=0.20
    export DENSITY_TEMPERATURE_T_MIN=0.10
    export DENSITY_TEMPERATURE_T_MAX=0.30
    export DENSITY_EMBEDDING_SCOPE=evidence_query_mean_pool
    bash run_direct_answer_vqa_experiments.sh
  "
done

echo "[submit] complete tag=$RUN_TAG"
