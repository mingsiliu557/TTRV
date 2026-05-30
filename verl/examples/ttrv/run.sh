#!/usr/bin/env bash
set -Eeuo pipefail
#export VLLM_ATTENTION_BACKEND=XFORMERS
# ray stop
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$VERL_DIR"

# ------------------------------------------------------------
mkdir -p logs

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)



TASK="${TASK:-dtd_20}"                       # put the dataset folder name here
NO_GPU="${NO_GPU:-4}"
EPISODE="${EPISODE:-2}"
ADVANTAGE="${ADVANTAGE:-grpo}"

K="${K:-3}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-7524}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
N="${N:-1}" # greedy validation by default


DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-$NO_GPU}"
N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"

DATA_LOCAL_DIR="${DATA_LOCAL_DIR:-$VERL_DIR/data}" # change this to your local data directory

BACKBONE_PATH="${BACKBONE_PATH:-OpenGVLab/InternVL3-2B}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BACKBONE_SAFE=$(echo "$BACKBONE_PATH" | tr '/' '_')
MODEL="${TASK}-${BACKBONE_SAFE}"
EXPERIMENT="TTRL-Len@${K}k"
TTRL_REWARD_STYLE="${TTRL_REWARD_STYLE:-frequency_entropy}"
SOFT_LABEL_GAMMA="${SOFT_LABEL_GAMMA:-2.0}"
UNKNOWN_REWARD="${UNKNOWN_REWARD:-0.0}"
ALL_UNKNOWN_REWARD="${ALL_UNKNOWN_REWARD:-0.0}"
ENTROPY_COEF="${ENTROPY_COEF:-0.75}"
ANSWER_PARSE_MODE="${ANSWER_PARSE_MODE:-legacy}"
HARMONY_TRANSFORM_TYPE="${HARMONY_TRANSFORM_TYPE:-photometric}"

WANDB_PROJECT="TTRL-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}"


LOG_FILE="logs/${TASK}_${BACKBONE_SAFE}_${EPISODE}e_${DATE}_${TIME_TAG}.log" # log file name

echo "[run] task=$TASK backbone=$BACKBONE_PATH gpus=$NO_GPU epochs=$EPISODE"
echo "[run] data=$DATA_LOCAL_DIR reward_style=$TTRL_REWARD_STYLE gamma=$SOFT_LABEL_GAMMA parser=$ANSWER_PARSE_MODE harmony_transform=$HARMONY_TRANSFORM_TYPE"
echo "[run] log=$LOG_FILE"

# see do_sample
# ------------------------------------------------------------
"$PYTHON_BIN" -m verl.trainer.main_ppo \
  reward_model.reward_manager=ttrl \
  reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  reward_model.reward_kwargs.mode="train" \
  reward_model.reward_kwargs.reward_style="$TTRL_REWARD_STYLE" \
  reward_model.reward_kwargs.soft_label_gamma="$SOFT_LABEL_GAMMA" \
  reward_model.reward_kwargs.unknown_reward="$UNKNOWN_REWARD" \
  reward_model.reward_kwargs.all_unknown_reward="$ALL_UNKNOWN_REWARD" \
  reward_model.reward_kwargs.entropy_coef="$ENTROPY_COEF" \
  reward_model.reward_kwargs.answer_parse_mode="$ANSWER_PARSE_MODE" \
  reward_model.reward_kwargs.harmony_transform_type="$HARMONY_TRANSFORM_TYPE" \
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
  actor_rollout_ref.rollout.do_vote=True \
  actor_rollout_ref.rollout.n_vote=$N_VOTES_PER_PROMPT \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.n=$N \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  trainer.logger=['console'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=$NO_GPU \
  trainer.nnodes=1 \
  trainer.save_freq=20000000 \
  trainer.test_freq=200000 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@" 2>&1 | tee "$LOG_FILE"
