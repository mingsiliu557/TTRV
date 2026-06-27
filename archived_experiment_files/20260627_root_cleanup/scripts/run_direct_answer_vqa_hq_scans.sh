#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-direct_answer_hq_scan_$(date +%Y%m%d_%H%M%S)}"
DATASETS=(${DATASETS:-visualsimpleqa ocrbench_v1 aokvqa_da_val})

run_one() {
  local key="$1"
  local task source final
  case "$key" in
    visualsimpleqa)
      task="visualsimpleqa_da_hq_pool80_seed42"
      source="visualsimpleqa_da"
      final="visualsimpleqa_da_20_hq_major_correct"
      ;;
    ocrbench_v1)
      task="ocrbench_v1_hq_pool80_seed42"
      source="ocrbench_v1"
      final="ocrbench_v1_20_hq_major_correct"
      ;;
    aokvqa_da_val)
      task="aokvqa_da_val_hq_pool80_seed42"
      source="aokvqa_da_val"
      final="aokvqa_da_val_20_hq_major_correct"
      ;;
    *)
      echo "unknown dataset key: $key" >&2
      exit 2
      ;;
  esac

  TASK="$task" \
  SOURCE_TASK="$source" \
  FINAL_TASK="$final" \
  RUN_TAG="${RUN_TAG}_${key}" \
  ANSWER_PARSE_MODE=direct_answer_short \
  ANSWER_CHOICE_LABELS=A-D \
  HQ_METRIC=direct_answer \
  bash run_hq_majority_scan.sh
}

for key in "${DATASETS[@]}"; do
  run_one "$key"
done

echo "[direct-answer-hq] complete tag=$RUN_TAG"
