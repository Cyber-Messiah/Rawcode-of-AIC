#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/aic_locate_rgb/Eagle/Embodied
source /root/autodl-tmp/venvs/locate_rgb/bin/activate

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/root/autodl-tmp/aic_locate_rgb/Eagle/Embodied:${PYTHONPATH:-}
export LAUNCHER=pytorch

RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
MAX_STEPS="${MAX_STEPS:-1000}"
FUSION_INIT="${FUSION_INIT:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"
OUTPUT_DIR="/root/autodl-tmp/aic_locate_rgb/work_dirs/locany_rgb_t_${RUN_TAG}"
RECIPE="${RECIPE:-/root/autodl-tmp/aic_locate_rgb/experiments/rgbt_cross_attention/rgbt_train_100_recipe.json}"
MODEL_PATH="/root/autodl-tmp/hf-cache/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0"
MAX_SAMPLES_PER_PACK="${MAX_SAMPLES_PER_PACK:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"

EXTRA_ARGS=()
if [[ -n "${FUSION_INIT}" ]]; then
  EXTRA_ARGS+=(--modality_fusion_path "${FUSION_INIT}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

/root/autodl-tmp/venvs/locate_rgb/bin/python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${RECIPE}" \
  --overwrite_output_dir True \
  --attn_implementation sdpa \
  --causal_attn False \
  --freeze_llm True \
  --freeze_backbone True \
  --freeze_mlp True \
  --use_modality_fusion True \
  --fusion_bottleneck_dim 256 \
  --fusion_window_size 5 \
  --fusion_max_residual_scale 0.1 \
  --modality_dropout_prob 0.1 \
  --save_trainable_only True \
  --max_steps "${MAX_STEPS}" \
  --block_size 4 \
  --max_seq_length 4096 \
  --max_num_tokens_per_sample 4096 \
  --max_num_tokens 4096 \
  --max_samples_per_pack "${MAX_SAMPLES_PER_PACK}" \
  --packing_buffer_size 1 \
  --dataloader_num_workers "${DATALOADER_WORKERS}" \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 5000 \
  --save_total_limit 2 \
  --save_every_n_hours 0 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_steps 20 \
  --lr_scheduler_type constant_with_warmup \
  --max_grad_norm 1.0 \
  --logging_steps 10 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --report_to none \
  --mlp_connector_layers 2 \
  "${EXTRA_ARGS[@]}"