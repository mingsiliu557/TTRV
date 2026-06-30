#!/usr/bin/env bash
set -Eeuo pipefail
#export VLLM_ATTENTION_BACKEND=XFORMERS
# ray stop
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$VERL_DIR"

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)



TASK="${TASK:-dtd_20}"                       # put the dataset folder name here
NO_GPU="${NO_GPU:-4}"
EPISODE="${EPISODE:-2}"
ADVANTAGE="${ADVANTAGE:-grpo}"

K="${K:-3}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-7524}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}"
ROLLOUT_LIMIT_IMAGES="${ROLLOUT_LIMIT_IMAGES:-1}"
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
ANSWER_CHOICE_LABELS="${ANSWER_CHOICE_LABELS:-A-D}"
HARMONY_TRANSFORM_TYPE="${HARMONY_TRANSFORM_TYPE:-photometric}"
FEATURE_CENTER_HSR_ALPHA="${FEATURE_CENTER_HSR_ALPHA:-0.5}"
FEATURE_CENTER_HSR_BETA="${FEATURE_CENTER_HSR_BETA:-0.2}"
ENTROPY_TEMPERATURE_VERSION="${ENTROPY_TEMPERATURE_VERSION:-v3}"
ENTROPY_TEMPERATURE_TAU0="${ENTROPY_TEMPERATURE_TAU0:-0.25}"
ENTROPY_TEMPERATURE_GAMMA="${ENTROPY_TEMPERATURE_GAMMA:-1.0}"
ENTROPY_TEMPERATURE_LAMBDA="${ENTROPY_TEMPERATURE_LAMBDA:-0.5}"
ENTROPY_TEMPERATURE_TAU_MIN="${ENTROPY_TEMPERATURE_TAU_MIN:-0.05}"
DENSITY_TEMPERATURE_T0="${DENSITY_TEMPERATURE_T0:-0.2}"
DENSITY_TEMPERATURE_T_MIN="${DENSITY_TEMPERATURE_T_MIN:-0.05}"
DENSITY_TEMPERATURE_T_MAX="${DENSITY_TEMPERATURE_T_MAX:-0.8}"
DENSITY_EMBEDDING_SCOPE="${DENSITY_EMBEDDING_SCOPE:-response_mean_pool}"
DENSITY_EVIDENCE_TEMPLATE="${DENSITY_EVIDENCE_TEMPLATE:-}"
NUMERIC_KERNEL_SIGMA="${NUMERIC_KERNEL_SIGMA:-0.15}"
NUMERIC_TRIM_RATIO="${NUMERIC_TRIM_RATIO:-0.2}"
USE_PAPO_PRCP_LOSS="${USE_PAPO_PRCP_LOSS:-false}"
PAPO_VALID_ONLY="${PAPO_VALID_ONLY:-false}"
PAPO_KL_PRCP_COEF="${PAPO_KL_PRCP_COEF:-0.01}"
PAPO_KL_PRCP_CLIP="${PAPO_KL_PRCP_CLIP:-0.2}"
PAPO_ORI_ENTROPY_COEF="${PAPO_ORI_ENTROPY_COEF:-0.0}"
PAPO_MASK_PATCH_SIZE="${PAPO_MASK_PATCH_SIZE:-14}"
PAPO_MASK_PROB="${PAPO_MASK_PROB:-0.6}"
PAPO_MASK_TYPE="${PAPO_MASK_TYPE:-black}"
PAPO_MASK_STRATEGY="${PAPO_MASK_STRATEGY:-random}"
PAPO_GROUNDING_FILE="${PAPO_GROUNDING_FILE:-}"
PAPO_GROUNDING_DIRECTION="${PAPO_GROUNDING_DIRECTION:-evidence}"
PAPO_GROUNDING_BOX_DILATE="${PAPO_GROUNDING_BOX_DILATE:-0.0}"
PAPO_FALLBACK_MASK="${PAPO_FALLBACK_MASK:-resolution}"
PAPO_FALLBACK_DOWNSCALE="${PAPO_FALLBACK_DOWNSCALE:-0.25}"
ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF:-0.001}"
ACTOR_CLIP_RATIO_LOW="${ACTOR_CLIP_RATIO_LOW:-0.2}"
ACTOR_CLIP_RATIO_HIGH="${ACTOR_CLIP_RATIO_HIGH:-0.2}"
TTRL_VAL_NUM_EXAMINE="${TTRL_VAL_NUM_EXAMINE:-0}"

WANDB_PROJECT="TTRL-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}"


LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${TASK}_${BACKBONE_SAFE}_${EPISODE}e_${DATE}_${TIME_TAG}.log}" # log file name

echo "[run] task=$TASK backbone=$BACKBONE_PATH gpus=$NO_GPU epochs=$EPISODE"
echo "[run] data=$DATA_LOCAL_DIR reward_style=$TTRL_REWARD_STYLE gamma=$SOFT_LABEL_GAMMA parser=$ANSWER_PARSE_MODE labels=$ANSWER_CHOICE_LABELS harmony_transform=$HARMONY_TRANSFORM_TYPE fc_hsr_alpha=$FEATURE_CENTER_HSR_ALPHA fc_hsr_beta=$FEATURE_CENTER_HSR_BETA"
echo "[run] entropy_temperature version=$ENTROPY_TEMPERATURE_VERSION tau0=$ENTROPY_TEMPERATURE_TAU0 gamma=$ENTROPY_TEMPERATURE_GAMMA lambda=$ENTROPY_TEMPERATURE_LAMBDA tau_min=$ENTROPY_TEMPERATURE_TAU_MIN"
echo "[run] density_temperature t0=$DENSITY_TEMPERATURE_T0 t_min=$DENSITY_TEMPERATURE_T_MIN t_max=$DENSITY_TEMPERATURE_T_MAX embedding_scope=$DENSITY_EMBEDDING_SCOPE log_embeddings=${TTRL_LOG_RESPONSE_EMBEDDINGS:-0}"
echo "[run] numeric_reward kernel_sigma=$NUMERIC_KERNEL_SIGMA trim_ratio=$NUMERIC_TRIM_RATIO"
echo "[run] papo use=$USE_PAPO_PRCP_LOSS valid_only=$PAPO_VALID_ONLY coef=$PAPO_KL_PRCP_COEF clip=$PAPO_KL_PRCP_CLIP entropy_coef=$PAPO_ORI_ENTROPY_COEF mask_strategy=$PAPO_MASK_STRATEGY mask_type=$PAPO_MASK_TYPE mask_patch=$PAPO_MASK_PATCH_SIZE mask_prob=$PAPO_MASK_PROB grounding_file=${PAPO_GROUNDING_FILE:-unset} grounding_direction=$PAPO_GROUNDING_DIRECTION fallback=$PAPO_FALLBACK_MASK"
echo "[run] actor kl_loss_coef=$ACTOR_KL_LOSS_COEF clip_low=$ACTOR_CLIP_RATIO_LOW clip_high=$ACTOR_CLIP_RATIO_HIGH"
echo "[run] rollout chunked_prefill=$ROLLOUT_ENABLE_CHUNKED_PREFILL limit_images=${ROLLOUT_LIMIT_IMAGES:-unset}"
echo "[run] log=$LOG_FILE"

ROLLOUT_LIMIT_IMAGES_ARG=()
if [[ -n "$ROLLOUT_LIMIT_IMAGES" ]]; then
  ROLLOUT_LIMIT_IMAGES_ARG=(+actor_rollout_ref.rollout.limit_images="$ROLLOUT_LIMIT_IMAGES")
fi

# see do_sample
# ------------------------------------------------------------
"$PYTHON_BIN" -m verl.trainer.main_ppo \
  reward_model.reward_manager=ttrl \
  +reward_model.val_num_examine=$TTRL_VAL_NUM_EXAMINE \
  reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  reward_model.reward_kwargs.mode="train" \
  reward_model.reward_kwargs.reward_style="$TTRL_REWARD_STYLE" \
  reward_model.reward_kwargs.soft_label_gamma="$SOFT_LABEL_GAMMA" \
  reward_model.reward_kwargs.unknown_reward="$UNKNOWN_REWARD" \
  reward_model.reward_kwargs.all_unknown_reward="$ALL_UNKNOWN_REWARD" \
  reward_model.reward_kwargs.entropy_coef="$ENTROPY_COEF" \
  reward_model.reward_kwargs.answer_parse_mode="$ANSWER_PARSE_MODE" \
  reward_model.reward_kwargs.answer_choice_labels="$ANSWER_CHOICE_LABELS" \
  reward_model.reward_kwargs.harmony_transform_type="$HARMONY_TRANSFORM_TYPE" \
  reward_model.reward_kwargs.feature_center_hsr_alpha="$FEATURE_CENTER_HSR_ALPHA" \
  reward_model.reward_kwargs.feature_center_hsr_beta="$FEATURE_CENTER_HSR_BETA" \
  reward_model.reward_kwargs.entropy_temperature_version="$ENTROPY_TEMPERATURE_VERSION" \
  reward_model.reward_kwargs.entropy_temperature_tau0="$ENTROPY_TEMPERATURE_TAU0" \
  reward_model.reward_kwargs.entropy_temperature_gamma="$ENTROPY_TEMPERATURE_GAMMA" \
  reward_model.reward_kwargs.entropy_temperature_lambda="$ENTROPY_TEMPERATURE_LAMBDA" \
  reward_model.reward_kwargs.entropy_temperature_tau_min="$ENTROPY_TEMPERATURE_TAU_MIN" \
  +reward_model.reward_kwargs.density_temperature_t0="$DENSITY_TEMPERATURE_T0" \
  +reward_model.reward_kwargs.density_temperature_t_min="$DENSITY_TEMPERATURE_T_MIN" \
  +reward_model.reward_kwargs.density_temperature_t_max="$DENSITY_TEMPERATURE_T_MAX" \
  +reward_model.reward_kwargs.density_embedding_scope="$DENSITY_EMBEDDING_SCOPE" \
  +reward_model.reward_kwargs.density_evidence_template="$DENSITY_EVIDENCE_TEMPLATE" \
  +reward_model.reward_kwargs.numeric_kernel_sigma="$NUMERIC_KERNEL_SIGMA" \
  +reward_model.reward_kwargs.numeric_trim_ratio="$NUMERIC_TRIM_RATIO" \
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.return_full_prompt=$USE_PAPO_PRCP_LOSS \
  data.papo_mask_patch_size=$PAPO_MASK_PATCH_SIZE \
  data.papo_mask_prob=$PAPO_MASK_PROB \
  +data.papo_mask_type=$PAPO_MASK_TYPE \
  data.papo_mask_strategy=$PAPO_MASK_STRATEGY \
  data.papo_grounding_file="$PAPO_GROUNDING_FILE" \
  data.papo_grounding_direction=$PAPO_GROUNDING_DIRECTION \
  data.papo_grounding_box_dilate=$PAPO_GROUNDING_BOX_DILATE \
  data.papo_fallback_mask=$PAPO_FALLBACK_MASK \
  data.papo_fallback_downscale=$PAPO_FALLBACK_DOWNSCALE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$ACTOR_KL_LOSS_COEF \
  actor_rollout_ref.actor.clip_ratio_low=$ACTOR_CLIP_RATIO_LOW \
  actor_rollout_ref.actor.clip_ratio_high=$ACTOR_CLIP_RATIO_HIGH \
  actor_rollout_ref.actor.use_papo_prcp_loss=$USE_PAPO_PRCP_LOSS \
  actor_rollout_ref.actor.papo_valid_only=$PAPO_VALID_ONLY \
  actor_rollout_ref.actor.papo_kl_prcp_coef=$PAPO_KL_PRCP_COEF \
  actor_rollout_ref.actor.papo_kl_prcp_clip=$PAPO_KL_PRCP_CLIP \
  actor_rollout_ref.actor.papo_ori_entropy_coef=$PAPO_ORI_ENTROPY_COEF \
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
  actor_rollout_ref.rollout.enable_chunked_prefill=$ROLLOUT_ENABLE_CHUNKED_PREFILL \
  "${ROLLOUT_LIMIT_IMAGES_ARG[@]}" \
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
