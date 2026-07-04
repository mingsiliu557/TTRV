#!/usr/bin/env bash
set -Eeuo pipefail

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
REPO_ROOT="/vepfs_default/chanxueyan/lhp/lms/TTRV"
EXP_ROOT="/jiigan-hp/ttrv-datasets/experiments/attention_space_papo_ai2d_softscale_${RUN_TAG}"
QUEUE_ID="${QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"

mkdir -p "$EXP_ROOT"
LAUNCH_LOG="$EXP_ROOT/volc_launch.log"

run_softscale() {
  local ratio="0.4"
  local ratio_tag="0p4"
  local scale="0.3"
  local scale_tag="0p3"
  local worker="$EXP_ROOT/worker_softscale_ratio_${ratio_tag}_scale_${scale_tag}.sh"

  cat > "$worker" <<'WORKER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/vepfs_default/chanxueyan/lhp/lms/TTRV"
VERL_DIR="$REPO_ROOT/verl"
DATA_ROOT="/jiigan-hp/ttrv-datasets"
EXP_ROOT="__EXP_ROOT__"
RATIO="__RATIO__"
RATIO_TAG="__RATIO_TAG__"
SCALE="__SCALE__"
SCALE_TAG="__SCALE_TAG__"
RUNTIME_ROOT="/vepfs_default/chanxueyan/lhp/lms/ttrv_runtime"
RUN_TAG="$(basename "$EXP_ROOT")"

mkdir -p "$EXP_ROOT"
exec > >(tee -a "$EXP_ROOT/worker_softscale_ratio_${RATIO_TAG}_scale_${SCALE_TAG}.log") 2>&1
cd "$VERL_DIR"

export PATH="/vepfs_default/chanxueyan/lhp/lms/envs/ttrv/bin:$PATH"
export PYTHON_BIN="/vepfs_default/chanxueyan/lhp/lms/envs/ttrv/bin/python"
export CUDA_VISIBLE_DEVICES="0,1"
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

export EPISODE=2
export DATA_TRAIN_BATCH_SIZE=2
export MINI_BATCH_SIZE=1
export MICRO_BATCH_SIZE=2
export N_VOTES_PER_PROMPT=32
export N_SAMPLES_PER_PROMPT=16
export N=1
export ENTROPY_COEF=0.0
export USE_PAPO_PRCP_LOSS=true
export PAPO_VALID_ONLY=false
export PAPO_KL_PRCP_CLIP=0.2
export PAPO_MASK_PATCH_SIZE=14
export PAPO_MASK_TYPE=black
export PAPO_CF_MODE=attention
export PAPO_ATTN_CF_RATIO="$RATIO"
export PAPO_ATTN_CF_CUT_IV=false
export PAPO_ATTN_CF_STYLE=soft_scale
export PAPO_ATTN_CF_SCALE="$SCALE"
export TTRL_VAL_NUM_EXAMINE=0
export TTRL_LOG_RESPONSE_EMBEDDINGS=0
export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
export ROLLOUT_LIMIT_IMAGES=1

key="ai2d20_softscale_ratio${RATIO_TAG}_scale${SCALE_TAG}"
run_name="attn_space_${key}_ori0_${RUN_TAG}"
run_dir="$EXP_ROOT/$key"
tmp_root="/tmp/as_ai2d_softscale_${RATIO_TAG}_${SCALE_TAG}_$$"
rm -rf "$tmp_root"
mkdir -p "$run_dir" "$tmp_root/t" "$tmp_root/r" "$tmp_root/tr" "$tmp_root/cu"
rm -f "$run_dir/SUCCESS" "$run_dir/FAILED"

export TASK="ai2d_20"
export ANSWER_PARSE_MODE="legacy"
export ANSWER_CHOICE_LABELS="A-D"
export TTRL_REWARD_STYLE="density_cluster_answer_entropy"
export DENSITY_EMBEDDING_SCOPE="evidence_query_mean_pool"
export DENSITY_TEMPERATURE_T0="0.20"
export DENSITY_TEMPERATURE_T_MIN="0.10"
export DENSITY_TEMPERATURE_T_MAX="0.30"
export PAPO_MASK_PROB="0.60"
export PAPO_KL_PRCP_COEF="0.01"
export PAPO_ORI_ENTROPY_COEF="0.0"
export ACTOR_KL_LOSS_COEF="0.001"
export ACTOR_CLIP_RATIO_LOW="0.2"
export ACTOR_CLIP_RATIO_HIGH="0.2"
export LOG_DIR="$run_dir"
export LOG_FILE="$run_dir/run.log"
export TTRL_TRAIN_ROLLOUT_JSONL="$run_dir/train_rollouts.jsonl"
export TTRL_EVAL_OUTPUT_JSONL="$run_dir/validation_step10_20_flat.jsonl"
export TTRL_EVAL_GROUP_OUTPUT_JSONL="$run_dir/validation_groups.jsonl"
export TMPDIR="$tmp_root/t"
export RAY_TMPDIR="$tmp_root/r"
export TRITON_CACHE_DIR="$tmp_root/tr"
export CUDA_CACHE_PATH="$tmp_root/cu"

echo "[worker] host=$(hostname) start=$(date) exp_root=$EXP_ROOT"
echo "[worker] branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true) commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || true)"
nvidia-smi -L || true
echo "[worker] ===== start $run_name $(date) ====="
echo "[worker] task=$TASK reward=$TTRL_REWARD_STYLE scope=$DENSITY_EMBEDDING_SCOPE parser=$ANSWER_PARSE_MODE labels=$ANSWER_CHOICE_LABELS"
echo "[worker] density T0=$DENSITY_TEMPERATURE_T0 Tmin=$DENSITY_TEMPERATURE_T_MIN Tmax=$DENSITY_TEMPERATURE_T_MAX"
echo "[worker] papo cf=$PAPO_CF_MODE attn_ratio=$PAPO_ATTN_CF_RATIO attn_style=$PAPO_ATTN_CF_STYLE attn_scale=$PAPO_ATTN_CF_SCALE mask=$PAPO_MASK_PROB kl=$PAPO_KL_PRCP_COEF ori=$PAPO_ORI_ENTROPY_COEF actor_kl=$ACTOR_KL_LOSS_COEF clip=$ACTOR_CLIP_RATIO_LOW/$ACTOR_CLIP_RATIO_HIGH"
echo "[worker] run_dir=$run_dir"

bash examples/ttrv/run.sh \
  trainer.val_before_train=False \
  trainer.test_freq=10 \
  trainer.save_freq=-1 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.experiment_name="$run_name" \
  trainer.default_local_dir="$run_dir/checkpoints"

touch "$run_dir/SUCCESS"
echo "[worker] ===== done $run_name $(date) ====="
exit 0
WORKER_EOF

  sed -i "s#__EXP_ROOT__#$EXP_ROOT#g; s#__RATIO__#$ratio#g; s#__RATIO_TAG__#$ratio_tag#g; s#__SCALE__#$scale#g; s#__SCALE_TAG__#$scale_tag#g" "$worker"
  chmod +x "$worker"

  echo "[launcher] ===== softscale ratio=$ratio scale=$scale start $(date) =====" | tee -a "$LAUNCH_LOG"
  volc ml_devinstance launch --resource_queue_id "$QUEUE_ID" --flavor_id "$FLAVOR_ID" bash "$worker" 2>&1 | tee -a "$LAUNCH_LOG"
  echo "[launcher] ===== softscale ratio=$ratio scale=$scale end $(date) =====" | tee -a "$LAUNCH_LOG"
}

echo "[launcher] exp_root=$EXP_ROOT" | tee -a "$LAUNCH_LOG"
echo "[launcher] setting=ratio0.4 soft_scale0.3" | tee -a "$LAUNCH_LOG"
run_softscale
echo "[launcher] all done $(date)" | tee -a "$LAUNCH_LOG"
