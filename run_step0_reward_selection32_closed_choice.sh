#!/usr/bin/env bash
set -Eeuo pipefail

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="/vepfs_default/chanxueyan/lhp/lms/TTRV"
EXP_ROOT="${EXP_ROOT:-/jiigan-hp/ttrv-datasets/experiments/step0_reward_selection32_${RUN_TAG}}"
QUEUE_ID="${QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"
RUN_SET="${RUN_SET:-full}"

mkdir -p "$EXP_ROOT"
MANIFEST="$EXP_ROOT/manifest.tsv"
MANAGER_LOG="$EXP_ROOT/manager.log"
: > "$MANAGER_LOG"
echo -e "key\ttask\tparser\tlabels\treward\tscope\tt0\ttmin\ttmax\tcoef\tori\tcf_mode\tattn_ratio\tattn_style\tattn_scale\tactor_kl\tclip_low\tclip_high" > "$MANIFEST"
echo "[manager] exp_root=$EXP_ROOT" | tee -a "$MANAGER_LOG"
echo "[manager] run_set=$RUN_SET queue=$QUEUE_ID flavor=$FLAVOR_ID" | tee -a "$MANAGER_LOG"
echo "[manager] step0 selection-only sample_n=32; no training update; no shutdown" | tee -a "$MANAGER_LOG"

write_worker() {
  local key="$1"
  local task="$2"
  local parser="$3"
  local labels="$4"
  local reward="$5"
  local scope="$6"
  local t0="$7"
  local tmin="$8"
  local tmax="$9"
  local coef="${10}"
  local ori="${11}"
  local cf_mode="${12}"
  local attn_ratio="${13}"
  local attn_style="${14}"
  local attn_scale="${15}"
  local actor_kl="${16}"
  local clip_low="${17}"
  local clip_high="${18}"
  local worker="$EXP_ROOT/worker_${key}.sh"

  echo -e "${key}\t${task}\t${parser}\t${labels}\t${reward}\t${scope}\t${t0}\t${tmin}\t${tmax}\t${coef}\t${ori}\t${cf_mode}\t${attn_ratio}\t${attn_style}\t${attn_scale}\t${actor_kl}\t${clip_low}\t${clip_high}" >> "$MANIFEST"

  cat > "$worker" <<'WORKER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/vepfs_default/chanxueyan/lhp/lms/TTRV"
VERL_DIR="$REPO_ROOT/verl"
RUNTIME_ROOT="/vepfs_default/chanxueyan/lhp/lms/ttrv_runtime"
DATA_ROOT="/jiigan-hp/ttrv-datasets"
EXP_ROOT="__EXP_ROOT__"
KEY="__KEY__"
TASK_NAME="__TASK__"
PARSE_MODE="__PARSER__"
CHOICE_LABELS="__LABELS__"
REWARD_STYLE="__REWARD__"
EMBED_SCOPE="__SCOPE__"
TEMP_T0="__T0__"
TEMP_TMIN="__TMIN__"
TEMP_TMAX="__TMAX__"
VIS_DEP_COEF="__COEF__"
ORI_ENTROPY="__ORI__"
CF_MODE="__CF_MODE__"
ATTN_RATIO="__ATTN_RATIO__"
ATTN_STYLE="__ATTN_STYLE__"
ATTN_SCALE="__ATTN_SCALE__"
ACTOR_KL="__ACTOR_KL__"
CLIP_LOW="__CLIP_LOW__"
CLIP_HIGH="__CLIP_HIGH__"
RUN_DIR="$EXP_ROOT/$KEY"
TMP_ROOT="/tmp/sel32_${KEY}_$$"

mkdir -p "$RUN_DIR" "$TMP_ROOT/t" "$TMP_ROOT/r" "$TMP_ROOT/tr" "$TMP_ROOT/cu"
trap 'status=$?; echo "[$KEY] FAILED status=$status end=$(date)"; touch "$RUN_DIR/FAILED"; ray stop --force || true; exit $status' ERR
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
cd "$VERL_DIR"

export PATH="/vepfs_default/chanxueyan/lhp/lms/envs/ttrv/bin:$PATH"
export PYTHON_BIN="/vepfs_default/chanxueyan/lhp/lms/envs/ttrv/bin/python"
export CUDA_VISIBLE_DEVICES=0,1
export NO_GPU=2
export DATA_LOCAL_DIR="$DATA_ROOT/verl_data"
export BACKBONE_PATH="$RUNTIME_ROOT/models/OpenGVLab/InternVL3-2B"
export HF_HOME="$RUNTIME_ROOT/hf_home"
export HF_MODULES_CACHE="$RUNTIME_ROOT/hf_home/modules"
export TRANSFORMERS_CACHE="$RUNTIME_ROOT/hf_home/transformers"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0

export TASK="$TASK_NAME"
export EPISODE=2
export DATA_TRAIN_BATCH_SIZE=2
export MINI_BATCH_SIZE=1
export MICRO_BATCH_SIZE=2
export N_VOTES_PER_PROMPT=32
export N_SAMPLES_PER_PROMPT=16
export N=32
export ANSWER_PARSE_MODE="$PARSE_MODE"
export ANSWER_CHOICE_LABELS="$CHOICE_LABELS"
export TTRL_REWARD_STYLE="$REWARD_STYLE"
export ENTROPY_COEF=0.0
export DENSITY_EMBEDDING_SCOPE="$EMBED_SCOPE"
export DENSITY_TEMPERATURE_T0="$TEMP_T0"
export DENSITY_TEMPERATURE_T_MIN="$TEMP_TMIN"
export DENSITY_TEMPERATURE_T_MAX="$TEMP_TMAX"
export TTRL_SELECTION_EVAL=1
export TTRL_VAL_NUM_EXAMINE=0
export TTRL_LOG_RESPONSE_EMBEDDINGS=0

export USE_PAPO_PRCP_LOSS=true
export PAPO_VALID_ONLY=false
export PAPO_KL_PRCP_COEF=0.0
export PAPO_KL_PRCP_CLIP=0.2
export PAPO_ORI_ENTROPY_COEF="$ORI_ENTROPY"
export PAPO_MASK_PATCH_SIZE=14
export PAPO_MASK_PROB=0.6
export PAPO_MASK_TYPE=black
export PAPO_CF_MODE="$CF_MODE"
export PAPO_ATTN_CF_RATIO="$ATTN_RATIO"
export PAPO_ATTN_CF_CUT_IV=false
export PAPO_ATTN_CF_STYLE="$ATTN_STYLE"
export PAPO_ATTN_CF_SCALE="$ATTN_SCALE"
export USE_VISUAL_DEP_REWARD=true
export VISUAL_DEP_VALID_ONLY=true
export VISUAL_DEP_COEF="$VIS_DEP_COEF"
export VISUAL_DEP_RAW_CLIP=20.0
export VISUAL_DEP_Z_CLIP=3.0
export ACTOR_KL_LOSS_COEF="$ACTOR_KL"
export ACTOR_CLIP_RATIO_LOW="$CLIP_LOW"
export ACTOR_CLIP_RATIO_HIGH="$CLIP_HIGH"
export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
export ROLLOUT_LIMIT_IMAGES=1

export LOG_DIR="$RUN_DIR"
export LOG_FILE="$RUN_DIR/run.log.inner"
export TTRL_EVAL_OUTPUT_JSONL="$RUN_DIR/validation_flat.jsonl"
export TTRL_EVAL_GROUP_OUTPUT_JSONL="$RUN_DIR/selection_groups.jsonl"
export TMPDIR="$TMP_ROOT/t"
export RAY_TMPDIR="$TMP_ROOT/r"
export TRITON_CACHE_DIR="$TMP_ROOT/tr"
export CUDA_CACHE_PATH="$TMP_ROOT/cu"

printf '[%s] host=%s start=%s branch=%s commit=%s\n' "$KEY" "$(hostname)" "$(date)" "$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)" "$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
echo "[$KEY] task=$TASK parser=$ANSWER_PARSE_MODE labels=$ANSWER_CHOICE_LABELS reward=$TTRL_REWARD_STYLE scope=$DENSITY_EMBEDDING_SCOPE"
echo "[$KEY] selection eval n=32 do_sample=true temp=1.0 top_p=0.95 visual_dep_coef=$VISUAL_DEP_COEF cf=$PAPO_CF_MODE/$PAPO_ATTN_CF_STYLE ratio=$PAPO_ATTN_CF_RATIO scale=$PAPO_ATTN_CF_SCALE"
echo "[$KEY] run_dir=$RUN_DIR"
nvidia-smi -L || true

bash examples/ttrv/run.sh \
  trainer.val_before_train=True \
  trainer.test_freq=0 \
  trainer.save_freq=0 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.total_epochs=0 \
  trainer.experiment_name="step0_reward_selection32_${KEY}" \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=32 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95

"$PYTHON_BIN" "$REPO_ROOT/verl/scripts/summarize_selection_groups.py" \
  --groups "$RUN_DIR/selection_groups.jsonl" \
  --summary-json "$RUN_DIR/summary.json" \
  --summary-md "$RUN_DIR/summary.md" \
  --dataset "$KEY"

ray stop --force || true
rm -rf "$TMP_ROOT" || true
touch "$RUN_DIR/SUCCESS"
echo "[$KEY] SUCCESS end=$(date)"
exit 0
WORKER_EOF

  sed -i \
    -e "s#__EXP_ROOT__#$EXP_ROOT#g" \
    -e "s#__KEY__#$key#g" \
    -e "s#__TASK__#$task#g" \
    -e "s#__PARSER__#$parser#g" \
    -e "s#__LABELS__#$labels#g" \
    -e "s#__REWARD__#$reward#g" \
    -e "s#__SCOPE__#$scope#g" \
    -e "s#__T0__#$t0#g" \
    -e "s#__TMIN__#$tmin#g" \
    -e "s#__TMAX__#$tmax#g" \
    -e "s#__COEF__#$coef#g" \
    -e "s#__ORI__#$ori#g" \
    -e "s#__CF_MODE__#$cf_mode#g" \
    -e "s#__ATTN_RATIO__#$attn_ratio#g" \
    -e "s#__ATTN_STYLE__#$attn_style#g" \
    -e "s#__ATTN_SCALE__#$attn_scale#g" \
    -e "s#__ACTOR_KL__#$actor_kl#g" \
    -e "s#__CLIP_LOW__#$clip_low#g" \
    -e "s#__CLIP_HIGH__#$clip_high#g" \
    "$worker"
  chmod +x "$worker"
}

launch_worker() {
  local key="$1"
  local worker="$EXP_ROOT/worker_${key}.sh"
  local session="sel32_${key}_${RUN_TAG:9:6}"
  session="${session:0:45}"
  echo "[manager] launch $key session=$session worker=$worker at $(date)" | tee -a "$MANAGER_LOG"
  tmux new-session -d -s "$session" "volc ml_devinstance launch --resource_queue_id $QUEUE_ID --flavor_id $FLAVOR_ID zsh"

  local launched=0
  for _ in $(seq 1 720); do
    local pane
    pane="$(tmux capture-pane -pt "$session" -S -220 2>/dev/null || true)"
    if printf '%s\n' "$pane" | grep -q "Connect worker with ip"; then
      sleep 10
      tmux send-keys -t "$session" "bash $worker; exit" C-m
      sleep 3
      tmux send-keys -t "$session" C-m
      launched=1
      echo "[manager] command sent for $key at $(date)" | tee -a "$MANAGER_LOG"
      break
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "[manager] session ended before connect for $key" | tee -a "$MANAGER_LOG"
      break
    fi
    sleep 10
  done

  if [[ "$launched" != 1 ]]; then
    echo "[manager] FAILED to connect worker for $key at $(date)" | tee -a "$MANAGER_LOG"
    tmux kill-session -t "$session" 2>/dev/null || true
    return 1
  fi

  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done

  if [[ -f "$EXP_ROOT/$key/SUCCESS" ]]; then
    echo "[manager] SUCCESS $key at $(date)" | tee -a "$MANAGER_LOG"
  else
    echo "[manager] FAILED $key at $(date); see $EXP_ROOT/$key/run.log" | tee -a "$MANAGER_LOG"
    return 1
  fi
  sleep 20
}

run_one() {
  local key="$1"
  shift
  write_worker "$key" "$@"
  if ! launch_worker "$key"; then
    echo "[manager] continuing after failure: $key" | tee -a "$MANAGER_LOG"
  fi
}

run_mme() {
  run_one "mme20" "mme_20" "legacy" "AB" "density_cluster_answer_entropy" "canonical_evidence_query_mean_pool" "0.20" "0.10" "0.30" "0.02" "0.00" "attention" "0.60" "hard_cut" "0.30" "0.01" "0.2" "0.3"
}

run_remaining() {
  run_one "ai2d20" "ai2d_20" "legacy" "A-D" "density_cluster_answer_entropy" "evidence_query_mean_pool" "0.20" "0.10" "0.30" "0.03" "0.00" "attention" "0.40" "soft_scale" "0.30" "0.001" "0.2" "0.2"
  run_one "realworldqa20_hq" "realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652" "legacy" "A-D" "density_cluster_soft" "canonical_evidence_query_mean_pool" "0.20" "0.20" "0.20" "0.05" "0.00" "attention" "0.40" "soft_scale" "0.30" "0.01" "0.2" "0.3"
  run_one "seed20" "seed_20" "legacy" "A-D" "density_cluster_answer_entropy" "canonical_evidence_query_mean_pool" "0.20" "0.10" "0.30" "0.03" "0.03" "attention" "0.30" "soft_scale" "0.50" "0.01" "0.2" "0.3"
  run_one "mathvista20_hq" "mathvista_20_hq_train13_repeat20" "legacy" "A-F" "density_cluster_answer_entropy" "canonical_evidence_query_mean_pool" "0.20" "0.10" "0.30" "0.05" "0.03" "attention" "0.30" "soft_scale" "0.50" "0.001" "0.2" "0.2"
  run_one "mathverse20_fixed" "mathverse_20_choice_norm_promptfix" "mathvista_choice" "A-F" "density_cluster_soft" "evidence_query_mean_pool" "0.20" "0.20" "0.20" "0.05" "0.02" "attention" "0.40" "soft_scale" "0.30" "0.001" "0.2" "0.2"
}

case "$RUN_SET" in
  smoke|mme_only)
    run_mme
    ;;
  remaining)
    run_remaining
    ;;
  full)
    run_mme
    run_remaining
    ;;
  *)
    echo "[manager] unknown RUN_SET=$RUN_SET; expected smoke, mme_only, remaining, full" | tee -a "$MANAGER_LOG"
    exit 2
    ;;
esac

EXP_ROOT_FOR_SUMMARY="$EXP_ROOT" "/vepfs_default/chanxueyan/lhp/lms/envs/ttrv/bin/python" - <<'SUMMARY_PY'
import json
import os
from pathlib import Path
root = Path(os.environ['EXP_ROOT_FOR_SUMMARY'])
rows = []
for path in sorted(root.glob('*/summary.json')):
    try:
        rows.append(json.loads(path.read_text()))
    except Exception as exc:
        rows.append({'dataset': path.parent.name, 'error': str(exc)})
(root / 'summary_all.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
lines = ['# Step0 Reward Selection32 Summary', '', '| dataset | groups | single@1 | major@32 | reward_select@32 | pass@32 | agreement | invalid | reward>major | major>reward |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    if row.get('groups'):
        lines.append(f"| {row.get('dataset')} | {row.get('groups')} | {row.get('single_at_1_pct', 0):.4f} | {row.get('major_vote_at_32_pct', 0):.4f} | {row.get('reward_select_at_32_pct', 0):.4f} | {row.get('pass_at_32_pct', 0):.4f} | {row.get('major_reward_agreement_pct', 0):.4f} | {row.get('invalid_ratio_pct', 0):.4f} | {row.get('reward_beats_major_count', 0)} | {row.get('major_beats_reward_count', 0)} |")
    else:
        lines.append(f"| {row.get('dataset')} | 0 | - | - | - | - | - | - | - | - |")
(root / 'summary_all.md').write_text('\n'.join(lines) + '\n')
SUMMARY_PY

echo "[manager] done at $(date) exp_root=$EXP_ROOT" | tee -a "$MANAGER_LOG"
