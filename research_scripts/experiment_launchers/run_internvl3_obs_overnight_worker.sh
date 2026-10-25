#!/usr/bin/env bash
set -Eeuo pipefail

# One-allocation InternVL3-2B Observation worker.
#
# This worker is intentionally orchestration-only.  It reuses the existing
# veRL validation path and attention extraction utilities without modifying
# model, reward, parser, or training code.  It must run inside a two-GPU A800
# worker allocated by run_internvl3_obs_overnight_manager.sh.

EXP_ROOT="${1:?Usage: bash run_internvl3_obs_overnight_worker.sh EXP_ROOT RUN_TAG}"
RUN_TAG="${2:?Usage: bash run_internvl3_obs_overnight_worker.sh EXP_ROOT RUN_TAG}"

REPO_ROOT=/workspace/TTRV
VERL_DIR="$REPO_ROOT/verl"
RUNTIME_ROOT=/workspace/ttrv_runtime
PYTHON_BIN=/workspace/envs/ttrv/bin/python
RAY_BIN=/workspace/envs/ttrv/bin/ray
BASE_MODEL="$RUNTIME_ROOT/models/OpenGVLab/InternVL3-2B"

OBS1_DATA_ROOT="$REPO_ROOT/observation_visualizations_20260728/data"
OBS1_TASK=aokvqa_obs1_prove_selection32
OBS1_SOURCE_DIR="$OBS1_DATA_ROOT/$OBS1_TASK"
OBS1_RUNNER="$VERL_DIR/examples/ttrv/run.sh"
OBS1_SUMMARIZER="$VERL_DIR/scripts/summarize_selection_groups.py"

OBS2_SOURCE_DIR="$REPO_ROOT/observation_visualizations_20260728/obs2_entropy_shaping_replay"

ATTENTION_SCRIPT="$REPO_ROOT/attention_visualization/attention_viz.py"
OBS3_SCREEN_SCRIPT="$REPO_ROOT/attention_visualization/obs3_screen.py"
PROMPT_AUDIT_SCRIPT="$VERL_DIR/scripts/audit_internvl3_prompt_tokens.py"
OBS3_STATIC_DIR="$REPO_ROOT/attention_visualization/obs3_validation_attention_20260728"
OBS3_EXPANDED_CANDIDATES="$OBS3_STATIC_DIR/candidates_expanded_all.jsonl"
OBS3_EXPANDED_SUMMARY="$OBS3_STATIC_DIR/candidates_expanded_summary.json"
OBS3_SMOKE_CANDIDATES=/data/ttrv/experiments/internvl3_obs3_validation_attention_20260728/probe/exact_candidates.jsonl

STAGE_ROOT="$EXP_ROOT/stages"
INPUT_ROOT="$EXP_ROOT/inputs"
STATUS_TSV="$EXP_ROOT/status.tsv"
WORKER_LOG="$EXP_ROOT/worker.log"
TMP_ROOT="/tmp/iv3o_$$"
CURRENT_STAGE=bootstrap

mkdir -p "$EXP_ROOT" "$STAGE_ROOT" "$TMP_ROOT"
exec > >(tee -a "$WORKER_LOG") 2>&1

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

append_status() {
  local stage=$1
  local state=$2
  local detail=${3:-}
  printf '%s\t%s\t%s\t%s\n' "$(timestamp)" "$stage" "$state" "$detail" >> "$STATUS_TSV"
}

wait_gpus_idle() {
  local polls=${1:-120}
  local threshold_mib=${2:-2048}
  local max_used=0
  local attempt
  for attempt in $(seq 1 "$polls"); do
    max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk 'BEGIN {m=0} {if ($1+0>m) m=$1+0} END {print m}')
    if ((max_used < threshold_mib)); then
      printf '[worker] GPUs idle max_used=%sMiB threshold=%sMiB\n' "$max_used" "$threshold_mib"
      return 0
    fi
    sleep 5
  done
  printf '[worker] GPUs did not become idle max_used=%sMiB threshold=%sMiB\n' \
    "$max_used" "$threshold_mib" >&2
  return 1
}

worker_exit() {
  local rc=$?
  trap - EXIT
  set +e
  "$RAY_BIN" stop --force >/dev/null 2>&1
  if [[ "$rc" == 0 ]]; then
    touch "$EXP_ROOT/SUCCESS"
    append_status overall SUCCESS "all required stages complete"
    printf '[worker] SUCCESS end=%s exp_root=%s\n' "$(timestamp)" "$EXP_ROOT"
  else
    touch "$EXP_ROOT/FAILED"
    append_status overall FAILED "rc=$rc current_stage=$CURRENT_STAGE"
    printf '[worker] FAILED rc=%s current_stage=%s end=%s\n' \
      "$rc" "$CURRENT_STAGE" "$(timestamp)" >&2
  fi
  if [[ -n "$TMP_ROOT" && "$TMP_ROOT" == /tmp/iv3o_[0-9]* ]]; then
    rm -rf -- "$TMP_ROOT"
  fi
  exit "$rc"
}

trap worker_exit EXIT
trap 'exit 130' INT TERM HUP

run_stage() {
  local stage=$1
  shift
  local stage_dir="$STAGE_ROOT/$stage"
  local stage_log="$stage_dir/stage.log"

  if [[ -e "$stage_dir" ]]; then
    printf '[worker] refusing to overwrite an existing stage directory: %s\n' "$stage_dir" >&2
    return 2
  fi
  mkdir -p "$stage_dir"
  touch "$stage_dir/STARTED"
  CURRENT_STAGE="$stage"
  append_status "$stage" STARTED "log=$stage_log"
  printf '[stage:%s] start=%s\n' "$stage" "$(timestamp)"

  # Do not put the stage function itself on the left side of a pipeline:
  # bash suppresses errexit for such a function, which can let an early
  # smoke/preflight failure be hidden by a later successful command. Also
  # avoid process substitution here: long-lived runtimes may close inherited
  # /dev/fd handles between stages. Write directly to the stage log instead.
  local rc
  set +e
  (
    set -Eeuo pipefail
    "$@"
  ) >> "$stage_log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$stage_dir/exit_code"
  if [[ "$rc" == 0 ]]; then
    touch "$stage_dir/SUCCESS"
    append_status "$stage" SUCCESS "rc=0"
    printf '[stage:%s] SUCCESS end=%s\n' "$stage" "$(timestamp)"
    CURRENT_STAGE=between_stages
    return 0
  fi

  touch "$stage_dir/FAILED"
  append_status "$stage" FAILED "rc=$rc"
  printf '[stage:%s] FAILED rc=%s end=%s\n' "$stage" "$rc" "$(timestamp)" >&2
  return "$rc"
}

skip_verified_stage() {
  local stage=$1
  shift
  local stage_dir="$STAGE_ROOT/$stage"
  local stage_log="$stage_dir/stage.log"

  if [[ -e "$stage_dir" ]]; then
    printf '[worker] refusing to overwrite an existing stage directory: %s\n' "$stage_dir" >&2
    return 2
  fi
  mkdir -p "$stage_dir"
  touch "$stage_dir/STARTED"
  CURRENT_STAGE="$stage"
  append_status "$stage" STARTED "verifying reusable result"

  local rc
  set +e
  (
    set -Eeuo pipefail
    "$@"
  ) >> "$stage_log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$stage_dir/exit_code"
  if [[ "$rc" == 0 ]]; then
    touch "$stage_dir/SKIPPED"
    append_status "$stage" SKIPPED "existing complete result reused"
    printf '[stage:%s] SKIPPED verified_existing_result end=%s\n' "$stage" "$(timestamp)"
    CURRENT_STAGE=between_stages
    return 0
  fi

  touch "$stage_dir/FAILED"
  append_status "$stage" FAILED "reuse verification rc=$rc"
  printf '[stage:%s] FAILED reusable result did not pass verification rc=%s\n' \
    "$stage" "$rc" >&2
  return "$rc"
}

preflight() {
  command -v nvidia-smi >/dev/null
  command -v git >/dev/null
  [[ -x "$PYTHON_BIN" ]]
  [[ -x "$RAY_BIN" ]]
  [[ -s "$BASE_MODEL/config.json" ]]
  [[ -s "$OBS1_SOURCE_DIR/train.parquet" ]]
  [[ -s "$OBS1_SOURCE_DIR/test.parquet" ]]
  [[ -s "$OBS1_RUNNER" ]]
  [[ -s "$OBS1_SUMMARIZER" ]]
  [[ -s "$ATTENTION_SCRIPT" ]]
  [[ -s "$OBS3_SCREEN_SCRIPT" ]]
  [[ -s "$PROMPT_AUDIT_SCRIPT" ]]
  [[ -s "$OBS3_EXPANDED_CANDIDATES" ]]
  [[ -s "$OBS3_EXPANDED_SUMMARY" ]]
  [[ -s "$OBS3_SMOKE_CANDIDATES" ]]

  local gpu_count
  gpu_count=$(nvidia-smi -L | wc -l)
  if ((gpu_count < 2)); then
    printf '[preflight] expected at least two visible GPUs, got %s\n' "$gpu_count" >&2
    return 2
  fi
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader

  "$PYTHON_BIN" - "$BASE_MODEL" "$OBS1_SOURCE_DIR" "$OBS2_SOURCE_DIR" \
    "$OBS3_EXPANDED_CANDIDATES" "$OBS3_EXPANDED_SUMMARY" \
    "$OBS3_SMOKE_CANDIDATES" "$INPUT_ROOT" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

base_model = Path(sys.argv[1]).resolve()
obs1 = Path(sys.argv[2]).resolve()
obs2 = Path(sys.argv[3]).resolve()
expanded = Path(sys.argv[4]).resolve()
expanded_summary_path = Path(sys.argv[5]).resolve()
smoke_candidates = Path(sys.argv[6]).resolve()
input_root = Path(sys.argv[7]).resolve()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]

assert torch.cuda.is_available(), 'PyTorch cannot see CUDA'
assert torch.cuda.device_count() >= 2, torch.cuda.device_count()
assert json.loads((base_model / 'config.json').read_text())['llm_config']['max_position_embeddings'] >= 8548
assert pq.ParquetFile(obs1 / 'train.parquet').metadata.num_rows == 20
assert pq.ParquetFile(obs1 / 'test.parquet').metadata.num_rows == 250

assert (obs2 / 'manager.exit_code').read_text().strip() == '0'
obs2_summary = json.loads((obs2 / 'run_summary.json').read_text())
assert obs2_summary['status'] == 'complete'
assert obs2_summary['cases'] == 9
assert obs2_summary['desired_effect_cases'] == 9
assert sum(1 for line in (obs2 / 'candidate_case_replay.jsonl').open() if line.strip()) == 9

summary = json.loads(expanded_summary_path.read_text())
assert summary['schema_version'] == 1
assert summary['status'] == 'PASS'
assert summary['counts'] == {'MME': 30, 'AI2D': 130, 'combined': 160}
for key in ('MME', 'AI2D', 'ALL'):
    source = Path(summary['outputs'][key]).resolve()
    assert source.is_file(), source
    assert digest(source) == summary['output_sha256'][key], key
assert expanded == Path(summary['outputs']['ALL']).resolve()

rows = read_jsonl(expanded)
assert len(rows) == 160
assert len({row['case_id'] for row in rows}) == 160
assert len({(row['dataset'], row['question_id']) for row in rows}) == 160
assert sum(row['dataset'] == 'MME' for row in rows) == 30
assert sum(row['dataset'] == 'AI2D' for row in rows) == 130
for row in rows:
    assert row['selection_gate'] == 'ours_correct_ttrv_valid_wrong'
    assert row['held_out_no_train_qid_or_image_overlap'] is True
    assert Path(row['image_path']).is_file(), row['image_path']
    assert row['retained_predictions']['ours']['parsed_answer'] == row['ground_truth']
    assert row['retained_predictions']['ttrv']['parsed_answer'] not in {row['ground_truth'], 'unknown'}
    for model_name in ('base', 'ttrv', 'ours'):
        checkpoint = Path(row['models'][model_name]['checkpoint'])
        assert (checkpoint / 'config.json').is_file(), checkpoint
        if model_name != 'base':
            assert (checkpoint / 'model.safetensors').is_file(), checkpoint
        assert Path(row['models'][model_name]['source_checkpoint']).exists()

smoke = read_jsonl(smoke_candidates)
assert len(smoke) == 1
assert smoke[0]['strict_eligible'] is True
for model_name in ('base', 'ttrv', 'ours'):
    checkpoint = Path(smoke[0]['models'][model_name]['checkpoint'])
    assert (checkpoint / 'config.json').is_file(), checkpoint

if input_root.exists():
    assert not any(input_root.iterdir()), f'input root is not empty: {input_root}'
else:
    input_root.mkdir(parents=True)
for key in ('MME', 'AI2D', 'ALL'):
    source = Path(summary['outputs'][key]).resolve()
    shutil.copy2(source, input_root / source.name)
shutil.copy2(expanded_summary_path, input_root / expanded_summary_path.name)
shutil.copy2(smoke_candidates, input_root / 'obs3_known_strict_smoke.jsonl')
validation = {
    'status': 'PASS',
    'obs1_rows': {'train': 20, 'test': 250},
    'obs2_reused_cases': 9,
    'obs3_rows': 160,
    'obs3_counts': summary['counts'],
    'obs3_sha256': summary['output_sha256'],
    'smoke_rows': 1,
}
(input_root / 'static_input_validation.json').write_text(json.dumps(validation, indent=2, sort_keys=True) + '\n')
print('[preflight] CUDA, Obs1, immutable Obs2, expanded Obs3, and known-strict smoke inputs passed')
PY

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$VERL_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$PROMPT_AUDIT_SCRIPT" \
      --obs1-dir "$OBS1_SOURCE_DIR" \
      --obs3-candidates "$INPUT_ROOT/candidates_expanded_all.jsonl" \
      --model "$BASE_MODEL" \
      --prompt-limit 7524 \
      --model-len 8548 \
      --obs1-max-new 1024 \
      --obs3-max-new 16 \
      --output-json "$INPUT_ROOT/prompt_token_audit.json"

  "$PYTHON_BIN" - "$INPUT_ROOT/prompt_token_audit.json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
assert payload['status'] == 'PASS'
assert payload['failures'] == []
assert payload['scopes']['obs1_train']['rows'] == 20
assert payload['scopes']['obs1_test']['rows'] == 250
assert payload['scopes']['obs3_candidates']['rows'] == 160
for scope in payload['scopes'].values():
    assert scope['over_prompt_limit'] == 0
    assert scope['over_context'] == 0
    assert scope['formula_runtime_match'] is True
    assert scope['patch_runtime_match'] is True
print('[preflight] processor-aware prompt-token audit passed for 20/250/160 rows')
PY

  {
    printf 'run_tag=%s\n' "$RUN_TAG"
    printf 'start=%s\n' "$(timestamp)"
    printf 'host=%s\n' "$(hostname)"
    printf 'repo=%s\n' "$REPO_ROOT"
    printf 'branch=%s\n' "$(git -C "$REPO_ROOT" branch --show-current)"
    printf 'commit=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf 'base_model=%s\n' "$BASE_MODEL"
    printf 'obs1_train_sha256=%s\n' "$(sha256sum "$OBS1_SOURCE_DIR/train.parquet" | awk '{print $1}')"
    printf 'obs1_test_sha256=%s\n' "$(sha256sum "$OBS1_SOURCE_DIR/test.parquet" | awk '{print $1}')"
    printf 'obs3_candidates_sha256=%s\n' "$(sha256sum "$INPUT_ROOT/candidates_expanded_all.jsonl" | awk '{print $1}')"
    printf 'obs3_candidates_count=160\n'
  } > "$EXP_ROOT/run_manifest.txt"
}

prepare_obs1_smoke_data() {
  local smoke_data_root=$1
  local smoke_task=$2
  local destination="$smoke_data_root/$smoke_task"
  mkdir -p "$destination"
  "$PYTHON_BIN" - "$OBS1_SOURCE_DIR" "$destination" <<'PY'
import sys
from pathlib import Path
from datasets import Dataset

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
for split, count in (("train", 2), ("test", 2)):
    dataset = Dataset.from_parquet(str(source / f"{split}.parquet"))
    dataset.select(range(count)).to_parquet(str(destination / f"{split}.parquet"))
print(f"[obs1-smoke] prepared 2 train and 2 test rows under {destination}")
PY
}

run_obs1_selection() {
  local task_name=$1
  local data_root=$2
  local run_dir=$3
  local sample_n=$4
  local expected_groups=$5
  local experiment_name=$6
  # Ray places AF_UNIX sockets below RAY_TMPDIR and Linux limits the socket
  # path to 107 bytes. Keep this independent of the descriptive experiment
  # name and fail locally if a future edit exceeds the budget.
  local tmp_dir="$TMP_ROOT/o${expected_groups}"
  local ray_socket_template="$tmp_dir/r/ray/session_2026-12-31_23-59-59_123456_123456/sockets/plasma_store"
  if ((${#ray_socket_template} > 107)); then
    printf '[obs1] Ray socket path budget exceeded: %s bytes (%s)\n' \
      "${#ray_socket_template}" "$ray_socket_template" >&2
    return 2
  fi
  printf '[obs1] Ray socket path budget=%s/107 tmp_dir=%s\n' \
    "${#ray_socket_template}" "$tmp_dir"

  mkdir -p "$run_dir" "$tmp_dir/t" "$tmp_dir/r" "$tmp_dir/tr" "$tmp_dir/cu"
  (
    set -Eeuo pipefail

    obs1_subprocess_exit() {
      local rc=$?
      local idle_rc=0
      trap - EXIT
      set +e
      "$RAY_BIN" stop --force >/dev/null 2>&1
      wait_gpus_idle 120 2048 || idle_rc=$?
      if [[ "$rc" == 0 && "$idle_rc" != 0 ]]; then
        rc=$idle_rc
      fi
      exit "$rc"
    }
    trap obs1_subprocess_exit EXIT

    unset LD_LIBRARY_PATH
    cd "$VERL_DIR"
    export PATH="/workspace/envs/ttrv/bin:$PATH"
    export PYTHON_BIN
    export CUDA_VISIBLE_DEVICES=0,1
    export NO_GPU=2
    export VLLM_USE_V1=0
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HOME="$RUNTIME_ROOT/hf_home"
    export HF_MODULES_CACHE="$RUNTIME_ROOT/hf_home/modules"
    export TRANSFORMERS_CACHE="$RUNTIME_ROOT/hf_home/transformers"
    export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg_cache"
    export BACKBONE_PATH="$BASE_MODEL"
    export DATA_LOCAL_DIR="$data_root"
    export TASK="$task_name"
    export EPISODE=2
    export DATA_TRAIN_BATCH_SIZE=2
    export MINI_BATCH_SIZE=1
    export MICRO_BATCH_SIZE=2
    export N_VOTES_PER_PROMPT=32
    export N_SAMPLES_PER_PROMPT=16
    export N="$sample_n"
    export MAX_PROMPT_LENGTH="${OBS1_MAX_PROMPT_LENGTH:-7524}"
    export MAX_RESPONSE_LENGTH="${OBS1_MAX_RESPONSE_LENGTH:-1024}"
    export ANSWER_PARSE_MODE=direct_answer_short
    export ANSWER_CHOICE_LABELS=A-D
    export TTRL_REWARD_STYLE=density_cluster_answer_entropy
    export DENSITY_EMBEDDING_SCOPE=evidence_query_mean_pool
    export DENSITY_TEMPERATURE_T0=0.20
    export DENSITY_TEMPERATURE_T_MIN=0.10
    export DENSITY_TEMPERATURE_T_MAX=0.30
    export ENTROPY_COEF=0.0
    export UNKNOWN_REWARD=0.0
    export ALL_UNKNOWN_REWARD=0.0
    export TTRL_SELECTION_EVAL=1
    export TTRL_VAL_NUM_EXAMINE=0
    export TTRL_LOG_RESPONSE_EMBEDDINGS=0
    export USE_PAPO_PRCP_LOSS=true
    export PAPO_VALID_ONLY=false
    export PAPO_KL_PRCP_COEF=0.0
    export PAPO_KL_PRCP_CLIP=0.2
    export PAPO_ORI_ENTROPY_COEF=0.0
    export PAPO_MASK_PATCH_SIZE=14
    export PAPO_MASK_PROB=0.60
    export PAPO_MASK_TYPE=black
    export PAPO_CF_MODE=pixel
    export USE_VISUAL_DEP_REWARD=true
    export VISUAL_DEP_VALID_ONLY=true
    export VISUAL_DEP_COEF=0.05
    export VISUAL_DEP_RAW_CLIP=20.0
    export VISUAL_DEP_Z_CLIP=3.0
    export ACTOR_KL_LOSS_COEF=0.01
    export ACTOR_CLIP_RATIO_LOW=0.2
    export ACTOR_CLIP_RATIO_HIGH=0.3
    export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
    export ROLLOUT_LIMIT_IMAGES=1
    export LOG_DIR="$run_dir"
    export LOG_FILE="$run_dir/run.log.inner"
    export TTRL_EVAL_OUTPUT_JSONL="$run_dir/validation_flat.jsonl"
    export TTRL_EVAL_GROUP_OUTPUT_JSONL="$run_dir/selection_groups.jsonl"
    export TMPDIR="$tmp_dir/t"
    export RAY_TMPDIR="$tmp_dir/r"
    export TRITON_CACHE_DIR="$tmp_dir/tr"
    export CUDA_CACHE_PATH="$tmp_dir/cu"

    printf '[obs1] task=%s expected_groups=%s n=%s start=%s\n' \
      "$TASK" "$expected_groups" "$N" "$(timestamp)"
    bash "$OBS1_RUNNER" \
      trainer.val_before_train=True \
      trainer.test_freq=0 \
      trainer.save_freq=0 \
      trainer.total_epochs=0 \
      data.filter_overlong_prompts_workers=4 \
      trainer.logger='[console]' \
      trainer.max_actor_ckpt_to_keep=0 \
      trainer.max_critic_ckpt_to_keep=0 \
      trainer.experiment_name="$experiment_name" \
      trainer.default_local_dir="$run_dir/checkpoints_disabled" \
      actor_rollout_ref.rollout.val_kwargs.do_sample=True \
      actor_rollout_ref.rollout.val_kwargs.n="$sample_n" \
      actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
      actor_rollout_ref.rollout.val_kwargs.top_p=0.95

    "$PYTHON_BIN" "$OBS1_SUMMARIZER" \
      --groups "$run_dir/selection_groups.jsonl" \
      --summary-json "$run_dir/summary.json" \
      --summary-md "$run_dir/summary.md" \
      --dataset "$task_name"

    "$PYTHON_BIN" - "$run_dir" "$expected_groups" "$sample_n" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected_groups = int(sys.argv[2])
expected_samples = int(sys.argv[3])
groups = [json.loads(line) for line in (run_dir / "selection_groups.jsonl").open() if line.strip()]
assert len(groups) == expected_groups, (len(groups), expected_groups)
assert all(len(group.get("samples") or []) == expected_samples for group in groups)
summary = json.loads((run_dir / "summary.json").read_text())
assert summary["groups"] == expected_groups
assert summary["samples"] == expected_groups * expected_samples
print(f"[obs1] integrity passed groups={expected_groups} samples_per_group={expected_samples}")
PY
  )
}

obs1_smoke() {
  local stage_dir="$STAGE_ROOT/01_integration_smoke_obs1"
  local smoke_data_root="$stage_dir/data"
  local smoke_task=obs1_prove_selection32_smoke
  prepare_obs1_smoke_data "$smoke_data_root" "$smoke_task"
  # Keep N=32 so the smoke exercises the same selection/reward grouping as
  # the full run; only the number of questions is reduced from 250 to 2.
  run_obs1_selection "$smoke_task" "$smoke_data_root" "$stage_dir/output" 32 2 \
    "internvl3_obs1_smoke_${RUN_TAG}"
}

obs3_smoke() {
  local stage_dir="$STAGE_ROOT/02_integration_smoke_obs3"
  local output_dir="$stage_dir/attention"
  local screen_dir="$stage_dir/screen"
  local candidates="$INPUT_ROOT/obs3_known_strict_smoke.jsonl"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$ATTENTION_SCRIPT" extract \
    --candidates "$candidates" \
    --output-dir "$output_dir" \
    --device cuda:0 \
    --max-new-tokens 16 \
    --limit 1 \
    --include-all \
    --save-individual-layers
  "$PYTHON_BIN" "$OBS3_SCREEN_SCRIPT" \
    --records "$output_dir/attention_records.jsonl" \
    --output-dir "$screen_dir" \
    --max-figures 1
  "$PYTHON_BIN" - "$output_dir/extraction_summary.json" \
    "$output_dir/attention_records.jsonl" "$screen_dir/screen_summary.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

extract = json.loads(Path(sys.argv[1]).read_text())
records = [json.loads(line) for line in Path(sys.argv[2]).open() if line.strip()]
screen = json.loads(Path(sys.argv[3]).read_text())
assert extract['requested_cases'] == 1
assert extract['successful_model_cases'] == 3
assert extract['failed_model_cases'] == 0
assert extract['complete_cases'] == 1
assert extract['save_individual_layers'] is True
assert len(records) == 3
for record in records:
    with np.load(record['npz_path']) as payload:
        for layer in range(28):
            assert f'first_decision_query__layer_{layer:02d}' in payload.files
assert screen['input_case_count'] == 1
assert screen['strict_complete_case_count'] == 1
assert screen['exclusion_counts'] == {}
print('[obs3-smoke] known strict case x Base/TTRV/Ours extraction and CPU screen passed')
PY
  wait_gpus_idle 120 2048
}

obs1_full() {
  local stage_dir="$STAGE_ROOT/10_obs1_prove_selection32"
  run_obs1_selection "$OBS1_TASK" "$OBS1_DATA_ROOT" "$stage_dir/output" 32 250 \
    "internvl3_obs1_prove_selection32_${RUN_TAG}"
}

verify_obs2_reuse() {
  local stage_dir="$STAGE_ROOT/20_obs2_replay_reused"
  "$PYTHON_BIN" - "$OBS2_SOURCE_DIR" "$stage_dir/reuse_manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
assert (source / "manager.exit_code").read_text().strip() == "0"
summary = json.loads((source / "run_summary.json").read_text())
assert summary["status"] == "complete"
assert summary["cases"] == 9
assert summary["desired_effect_cases"] == 9
replay = source / "candidate_case_replay.jsonl"
assert sum(1 for line in replay.open() if line.strip()) == 9

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

payload = {
    "state": "SKIPPED_REUSED",
    "reason": "The completed Obs2 GPU replay is immutable input to this overnight run.",
    "source": str(source),
    "manager_exit_code": 0,
    "summary": summary,
    "sha256": {
        "run_summary.json": digest(source / "run_summary.json"),
        "candidate_case_replay.jsonl": digest(replay),
        "candidate_case_summary.csv": digest(source / "candidate_case_summary.csv"),
    },
}
output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
}

obs3_full() {
  local stage_dir="$STAGE_ROOT/30_obs3_attention_visualization"
  local probe_dir="$stage_dir/probe"
  local attention_dir="$stage_dir/attention"
  local screen_dir="$stage_dir/screen"
  local candidates="$INPUT_ROOT/candidates_expanded_all.jsonl"

  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$ATTENTION_SCRIPT" probe \
    --candidates "$candidates" \
    --output-dir "$probe_dir" \
    --device cuda:0 \
    --max-new-tokens 16 \
    --include-all

  "$PYTHON_BIN" - "$probe_dir/probe_summary.json" <<'PY'
import json
import sys
from pathlib import Path
probe = json.loads(Path(sys.argv[1]).read_text())
assert probe['requested_cases'] == 160
assert probe['successful_model_cases'] == 480
assert probe['failed_model_cases'] == 0
assert probe['complete_cases'] == 160
assert Path(probe['exact_candidates']).is_file()
print(f"[obs3] probe integrity passed strict={probe['strict_eligible_cases']}")
PY

  local exact_candidates="$probe_dir/exact_candidates.jsonl"
  local exact_count
  exact_count=$(wc -l < "$exact_candidates")
  printf '[obs3] strict exact candidates=%s target_minimum=3\n' "$exact_count"
  if ((exact_count == 0)); then
    "$PYTHON_BIN" - "$probe_dir/probe_summary.json" "$stage_dir/result_summary.json" <<'PY'
import json
import sys
from pathlib import Path
probe = json.loads(Path(sys.argv[1]).read_text())
result = {
    'status': 'INSUFFICIENT_STRICT',
    'probed_cases': probe['requested_cases'],
    'strict_eligible_cases': 0,
    'target_strict_cases': 3,
    'gpu_artifacts_complete': False,
    'reason': 'No strict greedy-retained matches; the correctness gate was not relaxed.',
}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
PY
    printf '[obs3] zero strict candidates; preserving probe and stopping without relaxing gate\n' >&2
    return 4
  fi

  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$ATTENTION_SCRIPT" extract \
    --candidates "$exact_candidates" \
    --output-dir "$attention_dir" \
    --device cuda:0 \
    --max-new-tokens 16 \
    --include-all \
    --save-individual-layers

  "$PYTHON_BIN" "$OBS3_SCREEN_SCRIPT" \
    --records "$attention_dir/attention_records.jsonl" \
    --output-dir "$screen_dir" \
    --max-figures "${OBS3_MAX_FIGURES:-12}"

  "$PYTHON_BIN" - "$probe_dir/probe_summary.json" \
    "$attention_dir/extraction_summary.json" "$screen_dir/screen_summary.json" \
    "$stage_dir/result_summary.json" <<'PY'
import json
import sys
from pathlib import Path

probe = json.loads(Path(sys.argv[1]).read_text())
extract = json.loads(Path(sys.argv[2]).read_text())
screen = json.loads(Path(sys.argv[3]).read_text())
strict = int(probe['strict_eligible_cases'])
assert extract['requested_cases'] == strict
assert extract['successful_model_cases'] == 3 * strict
assert extract['failed_model_cases'] == 0
assert extract['complete_cases'] == strict
assert extract['save_individual_layers'] is True
assert screen['input_case_count'] == strict
assert screen['strict_complete_case_count'] == strict
assert screen['exclusion_counts'] == {}
assert Path(screen['ranked_cases']).is_file()
result = {
    'status': 'PASS' if strict >= 3 else 'INSUFFICIENT_STRICT',
    'probed_cases': probe['requested_cases'],
    'strict_eligible_cases': strict,
    'target_strict_cases': 3,
    'strict_target_met': strict >= 3,
    'extracted_complete_cases': extract['complete_cases'],
    'qualified_case_count': screen['qualified_case_count'],
    'rendered_case_count': screen['rendered_case_count'],
    'diagnostic_only': screen['diagnostic_only'],
    'selection_policy': screen['selection_policy'],
    'gpu_artifacts_complete': True,
    'gate_relaxed': False,
    'temperature_used_for_selection_or_ranking': False,
}
Path(sys.argv[4]).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(
    '[obs3] integrity passed '
    f"probed={probe['requested_cases']} strict={strict} "
    f"qualified={screen['qualified_case_count']} rendered={screen['rendered_case_count']}"
)
PY
  wait_gpus_idle 120 2048
  if ((exact_count < 3)); then
    printf '[obs3] strict target not met (%s < 3); all available GPU artifacts preserved\n' \
      "$exact_count" >&2
    return 5
  fi
}

if [[ -e "$EXP_ROOT/SUCCESS" || -e "$EXP_ROOT/FAILED" ]]; then
  printf '[worker] refusing to reuse a terminal experiment root: %s\n' "$EXP_ROOT" >&2
  exit 2
fi
if [[ -n "$(find "$STAGE_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf '[worker] refusing to overwrite nonempty stage root: %s\n' "$STAGE_ROOT" >&2
  exit 2
fi

printf 'timestamp\tstage\tstate\tdetail\n' > "$STATUS_TSV"
append_status overall STARTED "run_tag=$RUN_TAG"
printf '[worker] start=%s host=%s pid=%s run_tag=%s\n' \
  "$(timestamp)" "$(hostname)" "$$" "$RUN_TAG"
printf '[worker] exp_root=%s repo=%s\n' "$EXP_ROOT" "$REPO_ROOT"
printf '[worker] model=InternVL3-2B allocation=two_A800 execution=serial\n'
printf '[worker] no training checkpoints, no source checkpoint mutation, no shutdown command\n'

run_stage 00_preflight preflight
run_stage 01_integration_smoke_obs1 obs1_smoke
run_stage 02_integration_smoke_obs3 obs3_smoke
run_stage 10_obs1_prove_selection32 obs1_full
skip_verified_stage 20_obs2_replay_reused verify_obs2_reuse
run_stage 30_obs3_attention_visualization obs3_full

CURRENT_STAGE=complete
exit 0
