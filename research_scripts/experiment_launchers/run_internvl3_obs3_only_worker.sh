#!/usr/bin/env bash
set -Eeuo pipefail

# Single-GPU, Obs3-only retry worker.
#
# This worker intentionally reuses the frozen expanded-160 candidate snapshot
# from the 20260730 overnight run. It does not train, merge checkpoints, mutate
# model weights, relax the correctness gate, or reuse the failed stage directory.

EXP_ROOT="${1:?Usage: bash run_internvl3_obs3_only_worker.sh EXP_ROOT RUN_TAG}"
RUN_TAG="${2:?Usage: bash run_internvl3_obs3_only_worker.sh EXP_ROOT RUN_TAG}"

REPO_ROOT=/workspace/TTRV
PYTHON_BIN=/workspace/envs/ttrv/bin/python
BASE_MODEL=/workspace/ttrv_runtime/models/OpenGVLab/InternVL3-2B
ATTENTION_SCRIPT="$REPO_ROOT/attention_visualization/attention_viz.py"
SCREEN_SCRIPT="$REPO_ROOT/attention_visualization/obs3_screen.py"

SOURCE_RUN=/data/ttrv/experiments/internvl3_obs_overnight_20260730_041611
SOURCE_CANDIDATES="$SOURCE_RUN/inputs/candidates_expanded_all.jsonl"
SOURCE_PROMPT_AUDIT="$SOURCE_RUN/inputs/prompt_token_audit.json"
EXPECTED_CANDIDATE_SHA256=c3e140e66b69b1e52fa137cc7f77816b60e81b4714092584b285e17857de9127

STAGE_ROOT="$EXP_ROOT/stages"
INPUT_ROOT="$EXP_ROOT/inputs"
STATUS_TSV="$EXP_ROOT/status.tsv"
WORKER_LOG="$EXP_ROOT/worker.log"
TMP_ROOT="/tmp/iv3o3_${RUN_TAG}_$$"
CURRENT_STAGE=bootstrap

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export TMPDIR="$TMP_ROOT"

# Refuse every worker-owned path before the first mkdir, log append, trap, or
# marker write. The manager is allowed to have created only its own log/status
# files in an otherwise fresh root.
if [[ -e "$EXP_ROOT/SUCCESS" || -e "$EXP_ROOT/FAILED" ]]; then
  printf '[worker] refusing terminal experiment root: %s\n' "$EXP_ROOT" >&2
  exit 2
fi
for owned_path in "$STAGE_ROOT" "$INPUT_ROOT" "$STATUS_TSV" "$WORKER_LOG" \
  "$EXP_ROOT/run_manifest.txt" "$EXP_ROOT/code_snapshot"; do
  if [[ -e "$owned_path" ]]; then
    printf '[worker] refusing existing worker-owned path: %s\n' "$owned_path" >&2
    exit 2
  fi
done

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

worker_exit() {
  local rc=$?
  trap - EXIT
  set +e
  if [[ "$rc" == 0 ]]; then
    touch "$EXP_ROOT/SUCCESS"
    append_status overall SUCCESS "all required Obs3 stages complete"
    printf '[worker] SUCCESS end=%s exp_root=%s\n' "$(timestamp)" "$EXP_ROOT"
  else
    touch "$EXP_ROOT/FAILED"
    append_status overall FAILED "rc=$rc current_stage=$CURRENT_STAGE"
    printf '[worker] FAILED rc=%s current_stage=%s end=%s\n' \
      "$rc" "$CURRENT_STAGE" "$(timestamp)" >&2
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
    printf '[worker] refusing to overwrite existing stage directory: %s\n' "$stage_dir" >&2
    return 2
  fi
  mkdir -p "$stage_dir"
  touch "$stage_dir/STARTED"
  CURRENT_STAGE="$stage"
  append_status "$stage" STARTED "log=$stage_log"
  printf '[stage:%s] start=%s\n' "$stage" "$(timestamp)"

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

resource_snapshot() {
  printf '[resource] timestamp=%s\n' "$(timestamp)"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version \
    --format=csv,noheader
  printf '[resource] compute_apps_begin\n'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  printf '[resource] compute_apps_end\n'
  free -h
  df -h /dev/shm /tmp
  sed -n '1,8p' /proc/meminfo
  for path in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max \
    /sys/fs/cgroup/memory/memory.usage_in_bytes \
    /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [[ -r "$path" ]]; then
      printf '[resource] %s=' "$path"
      sed -n '1p' "$path"
    fi
  done
  ps -eo pid,ppid,stat,rss,etime,cmd
}

reject_residual_gpu_services() {
  if ps -eo pid,ppid,cmd | grep -Ei '[r]aylet|[g]cs_server|[v]llm' ; then
    printf '[resource] residual Ray/vLLM process detected on fresh worker\n' >&2
    return 2
  fi
}

wait_gpu_quiet() {
  local polls=${1:-12}
  local required_consecutive=${2:-2}
  local threshold_mib=${3:-512}
  local consecutive=0
  local attempt
  local used
  local compute_pids

  for attempt in $(seq 1 "$polls"); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk 'NR == 1 {print $1 + 0}')
    compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    printf '[resource] quiet_check attempt=%s used_mib=%s compute_pids=%s\n' \
      "$attempt" "$used" "${compute_pids:-none}"
    if ((used < threshold_mib)) && [[ -z "$compute_pids" ]]; then
      ((consecutive += 1))
      if ((consecutive >= required_consecutive)); then
        printf '[resource] GPU quiet passed consecutive=%s threshold_mib=%s\n' \
          "$consecutive" "$threshold_mib"
        return 0
      fi
    else
      consecutive=0
    fi
    sleep 5
  done
  printf '[resource] GPU quiet check failed used_mib=%s compute_pids=%s\n' \
    "$used" "${compute_pids:-none}" >&2
  return 2
}

check_memory_headroom() {
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path

gib = 1024 ** 3
host_required = 24 * gib
cgroup_required = 16 * gib
meminfo = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    if ":" in line:
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0]) * 1024
available = meminfo["MemAvailable"]
assert available >= host_required, (
    f"host MemAvailable below 24 GiB: {available / gib:.2f} GiB"
)

def check_cgroup_pair(current_path: Path, max_path: Path, label: str) -> None:
    if not (current_path.is_file() and max_path.is_file()):
        return
    current = int(current_path.read_text().strip())
    raw_max = max_path.read_text().strip()
    if raw_max == "max":
        print(f"[resource] {label}_headroom=unlimited")
        return
    maximum = int(raw_max)
    # Some cgroup-v1 deployments use a near-int64 sentinel for "unlimited".
    if maximum > meminfo["MemTotal"] * 2:
        print(f"[resource] {label}_headroom=unlimited_sentinel")
        return
    headroom = maximum - current
    assert headroom >= cgroup_required, (
        f"{label} headroom below 16 GiB: {headroom / gib:.2f} GiB"
    )
    print(f"[resource] {label}_headroom_gib={headroom / gib:.2f}")

check_cgroup_pair(
    Path("/sys/fs/cgroup/memory.current"),
    Path("/sys/fs/cgroup/memory.max"),
    "cgroup_v2",
)
check_cgroup_pair(
    Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    "cgroup_v1",
)
print(f"[resource] host_mem_available_gib={available / gib:.2f}")
PY
}

cuda_sanity() {
  "$PYTHON_BIN" - <<'PY'
import gc
import json
import torch

assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
before_free, before_total = torch.cuda.mem_get_info(device)
x = torch.ones((1024, 1024), dtype=torch.float32, device=device)
value = float(x.sum().item())
torch.cuda.synchronize(device)
after_free, after_total = torch.cuda.mem_get_info(device)
payload = {
    "device": props.name,
    "device_count": torch.cuda.device_count(),
    "tensor_sum": value,
    "free_before_gib": before_free / 1024**3,
    "free_after_gib": after_free / 1024**3,
    "total_gib": before_total / 1024**3,
}
print("[cuda-sanity] " + json.dumps(payload, sort_keys=True))
del x
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize(device)
PY
}

run_gpu_command() {
  local rc
  set +e
  "$@"
  rc=$?
  set -e
  if [[ "$rc" != 0 ]]; then
    printf '[resource] GPU command failed rc=%s; capturing failure-time state\n' "$rc" >&2
    resource_snapshot
    return "$rc"
  fi
}

preflight() {
  command -v nvidia-smi >/dev/null
  command -v git >/dev/null
  command -v free >/dev/null
  [[ -x "$PYTHON_BIN" ]]
  [[ -s "$ATTENTION_SCRIPT" ]]
  [[ -s "$SCREEN_SCRIPT" ]]
  [[ -s "$BASE_MODEL/config.json" ]]
  [[ -s "$SOURCE_CANDIDATES" ]]
  [[ -s "$SOURCE_PROMPT_AUDIT" ]]

  local gpu_count
  gpu_count=$(nvidia-smi -L | wc -l)
  if ((gpu_count != 1)); then
    printf '[preflight] expected exactly one physical GPU, got %s\n' "$gpu_count" >&2
    return 2
  fi

  resource_snapshot
  reject_residual_gpu_services
  wait_gpu_quiet 12 2 512
  check_memory_headroom
  cuda_sanity

  if [[ -e "$INPUT_ROOT" ]] && [[ -n "$(find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf '[preflight] refusing nonempty input root: %s\n' "$INPUT_ROOT" >&2
    return 2
  fi
  mkdir -p "$INPUT_ROOT"
  cp "$SOURCE_CANDIDATES" "$INPUT_ROOT/candidates_expanded_all.jsonl"
  cp "$SOURCE_PROMPT_AUDIT" "$INPUT_ROOT/prompt_token_audit.json"

  local snapshot_dir="$EXP_ROOT/code_snapshot"
  mkdir -p "$snapshot_dir"
  cp "$REPO_ROOT/research_scripts/experiment_launchers/run_internvl3_obs_overnight_manager.sh" "$snapshot_dir/"
  cp "$REPO_ROOT/research_scripts/experiment_launchers/run_internvl3_obs3_only_worker.sh" "$snapshot_dir/"
  cp "$ATTENTION_SCRIPT" "$snapshot_dir/"
  cp "$SCREEN_SCRIPT" "$snapshot_dir/"
  sha256sum "$snapshot_dir"/* > "$snapshot_dir/SHA256SUMS"

  "$PYTHON_BIN" - "$INPUT_ROOT/candidates_expanded_all.jsonl" \
    "$INPUT_ROOT/prompt_token_audit.json" "$INPUT_ROOT" \
    "$EXPECTED_CANDIDATE_SHA256" <<'PY'
import collections
import hashlib
import json
import sys
from pathlib import Path

candidates = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
input_root = Path(sys.argv[3])
expected_sha = sys.argv[4]

actual_sha = hashlib.sha256(candidates.read_bytes()).hexdigest()
assert actual_sha == expected_sha, (actual_sha, expected_sha)
rows = [json.loads(line) for line in candidates.open() if line.strip()]
assert len(rows) == 160
assert len({row["case_id"] for row in rows}) == 160
assert len({(row["dataset"], row["question_id"]) for row in rows}) == 160
counts = collections.Counter(row["dataset"] for row in rows)
assert counts == {"MME": 30, "AI2D": 130}, counts

checkpoints = set()
for row in rows:
    assert row["selection_gate"] == "ours_correct_ttrv_valid_wrong"
    assert row["held_out_no_train_qid_or_image_overlap"] is True
    assert Path(row["image_path"]).is_file(), row["image_path"]
    assert set(row["models"]) == {"base", "ttrv", "ours"}
    for model in ("base", "ttrv", "ours"):
        checkpoint = Path(row["models"][model]["checkpoint"])
        checkpoints.add(checkpoint)
        assert checkpoint.is_dir(), checkpoint
        assert (checkpoint / "config.json").is_file(), checkpoint
        has_weights = any(checkpoint.glob("*.safetensors")) or any(
            checkpoint.glob("pytorch_model*.bin")
        )
        assert has_weights, checkpoint
assert len(checkpoints) == 5, sorted(map(str, checkpoints))

audit = json.loads(audit_path.read_text())
obs3 = audit["scopes"]["obs3_candidates"]
assert audit["status"] == "PASS"
assert not audit["failures"]
assert obs3["status"] == "PASS"
assert obs3["rows"] == 160
assert obs3["max_new_tokens"] == 16
assert obs3["over_context"] == 0
assert obs3["over_prompt_limit"] == 0

smoke_case_ids = ("MME_372391d34608", "AI2D_13ab674095c7")
by_case_id = {row["case_id"]: row for row in rows}
smoke_rows = [by_case_id[case_id] for case_id in smoke_case_ids]
smoke_path = input_root / "cross_dataset_smoke.jsonl"
smoke_path.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in smoke_rows)
)

payload = {
    "status": "PASS",
    "candidate_rows": len(rows),
    "candidate_sha256": actual_sha,
    "dataset_counts": dict(counts),
    "checkpoint_groups": sorted(map(str, checkpoints)),
    "prompt_audit": {
        "status": obs3["status"],
        "max_prompt_tokens": obs3["max_prompt_tokens"],
        "max_total_tokens": obs3["max_total_tokens"],
        "model_len": obs3["model_len"],
    },
    "cross_dataset_smoke_case_ids": [row["case_id"] for row in smoke_rows],
}
(input_root / "static_input_validation.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print("[preflight] " + json.dumps(payload, sort_keys=True))
PY

  {
    printf 'run_tag=%s\n' "$RUN_TAG"
    printf 'start=%s\n' "$(timestamp)"
    printf 'host=%s\n' "$(hostname)"
    printf 'repo=%s\n' "$REPO_ROOT"
    printf 'branch=%s\n' "$(git -C "$REPO_ROOT" branch --show-current)"
    printf 'commit=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf 'base_model=%s\n' "$BASE_MODEL"
    printf 'source_run=%s\n' "$SOURCE_RUN"
    printf 'candidate_sha256=%s\n' "$EXPECTED_CANDIDATE_SHA256"
    printf 'candidate_count=160\n'
    printf 'allocation=one_A800\n'
    printf 'training=false\n'
    printf 'code_snapshot=%s\n' "$EXP_ROOT/code_snapshot"
    printf 'worker_sha256=%s\n' \
      "$(sha256sum "$REPO_ROOT/research_scripts/experiment_launchers/run_internvl3_obs3_only_worker.sh" | awk '{print $1}')"
    printf 'attention_viz_sha256=%s\n' \
      "$(sha256sum "$ATTENTION_SCRIPT" | awk '{print $1}')"
    printf 'obs3_screen_sha256=%s\n' \
      "$(sha256sum "$SCREEN_SCRIPT" | awk '{print $1}')"
  } > "$EXP_ROOT/run_manifest.txt"
}

cross_dataset_smoke() {
  local stage_dir="$STAGE_ROOT/01_cross_dataset_smoke"
  local output_dir="$stage_dir/attention"
  local candidates="$INPUT_ROOT/cross_dataset_smoke.jsonl"

  run_gpu_command env CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$ATTENTION_SCRIPT" extract \
    --candidates "$candidates" \
    --output-dir "$output_dir" \
    --device cuda:0 \
    --max-new-tokens 16 \
    --include-all \
    --save-individual-layers

  "$PYTHON_BIN" - "$output_dir/extraction_summary.json" \
    "$output_dir/attention_records.jsonl" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

summary = json.loads(Path(sys.argv[1]).read_text())
records = [json.loads(line) for line in Path(sys.argv[2]).open() if line.strip()]
assert summary["requested_cases"] == 2
assert summary["successful_model_cases"] == 6
assert summary["failed_model_cases"] == 0
assert summary["complete_cases"] == 2
assert summary["save_individual_layers"] is True
assert len(records) == 6
for record in records:
    with np.load(record["npz_path"]) as payload:
        for layer in range(28):
            assert f"first_decision_query__layer_{layer:02d}" in payload.files
print("[smoke] MME+AI2D x Base/TTRV/Ours and all 28 layers passed")
PY

  wait_gpu_quiet 24 2 512
  resource_snapshot
  check_memory_headroom
  cuda_sanity
}

full_probe() {
  local stage_dir="$STAGE_ROOT/10_full_probe"
  local probe_dir="$stage_dir/probe"
  local candidates="$INPUT_ROOT/candidates_expanded_all.jsonl"

  run_gpu_command env CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$ATTENTION_SCRIPT" probe \
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
assert probe["requested_cases"] == 160
assert probe["successful_model_cases"] == 480
assert probe["failed_model_cases"] == 0
assert probe["complete_cases"] == 160
exact_candidates = Path(probe["exact_candidates"])
assert exact_candidates.is_file()
exact_count = sum(1 for line in exact_candidates.open() if line.strip())
assert exact_count == probe["strict_eligible_cases"]
print(f"[probe] integrity passed strict={probe['strict_eligible_cases']}")
PY

  wait_gpu_quiet 24 2 512
  resource_snapshot
  check_memory_headroom
  cuda_sanity
}

extract_and_screen() {
  local probe_dir="$STAGE_ROOT/10_full_probe/probe"
  local stage_dir="$STAGE_ROOT/20_extract_and_screen"
  local attention_dir="$stage_dir/attention"
  local screen_dir="$stage_dir/screen"
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
    "status": "INSUFFICIENT_STRICT",
    "probed_cases": probe["requested_cases"],
    "strict_eligible_cases": 0,
    "target_strict_cases": 3,
    "strict_target_met": False,
    "gpu_artifacts_complete": False,
    "gate_relaxed": False,
    "temperature_used_for_selection_or_ranking": False,
    "reason": "No strict greedy-retained matches; the correctness gate was not relaxed.",
}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
PY
    cp "$stage_dir/result_summary.json" "$EXP_ROOT/result_summary.json"
    printf '[obs3] zero strict candidates; probe preserved without gate relaxation\n' >&2
    return 4
  fi

  run_gpu_command env CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$ATTENTION_SCRIPT" extract \
    --candidates "$exact_candidates" \
    --output-dir "$attention_dir" \
    --device cuda:0 \
    --max-new-tokens 16 \
    --include-all \
    --save-individual-layers

  "$PYTHON_BIN" "$SCREEN_SCRIPT" \
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
strict = int(probe["strict_eligible_cases"])
assert extract["requested_cases"] == strict
assert extract["successful_model_cases"] == 3 * strict
assert extract["failed_model_cases"] == 0
assert extract["complete_cases"] == strict
assert extract["save_individual_layers"] is True
assert screen["input_case_count"] == strict
assert screen["strict_complete_case_count"] == strict
assert screen["exclusion_counts"] == {}
assert Path(screen["ranked_cases"]).is_file()
result = {
    "status": "PASS" if strict >= 3 else "INSUFFICIENT_STRICT",
    "probed_cases": probe["requested_cases"],
    "strict_eligible_cases": strict,
    "target_strict_cases": 3,
    "strict_target_met": strict >= 3,
    "extracted_complete_cases": extract["complete_cases"],
    "qualified_case_count": screen["qualified_case_count"],
    "rendered_case_count": screen["rendered_case_count"],
    "diagnostic_only": screen["diagnostic_only"],
    "selection_policy": screen["selection_policy"],
    "gpu_artifacts_complete": True,
    "gate_relaxed": False,
    "temperature_used_for_selection_or_ranking": False,
}
Path(sys.argv[4]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(
    "[obs3] integrity passed "
    f"probed={probe['requested_cases']} strict={strict} "
    f"qualified={screen['qualified_case_count']} "
    f"rendered={screen['rendered_case_count']}"
)
PY
  cp "$stage_dir/result_summary.json" "$EXP_ROOT/result_summary.json"

  wait_gpu_quiet 24 2 512
  if ((exact_count < 3)); then
    printf '[obs3] strict target not met (%s < 3); all GPU artifacts preserved\n' \
      "$exact_count" >&2
    return 5
  fi
}

printf 'timestamp\tstage\tstate\tdetail\n' > "$STATUS_TSV"
append_status overall STARTED "run_tag=$RUN_TAG"
printf '[worker] start=%s host=%s pid=%s run_tag=%s\n' \
  "$(timestamp)" "$(hostname)" "$$" "$RUN_TAG"
printf '[worker] exp_root=%s repo=%s\n' "$EXP_ROOT" "$REPO_ROOT"
printf '[worker] model=InternVL3-2B allocation=one_A800 execution=Obs3_only\n'
printf '[worker] no training, no checkpoint mutation, no gate relaxation, no shutdown\n'

run_stage 00_preflight preflight
run_stage 01_cross_dataset_smoke cross_dataset_smoke
run_stage 10_full_probe full_probe
run_stage 20_extract_and_screen extract_and_screen

CURRENT_STAGE=complete
exit 0
