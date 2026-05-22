#!/bin/bash
# Eagle3 Online Draft Training + GRPO on a single 8-GPU node
# Trains Qwen3-1.7B with GRPO while co-training an Eagle3 draft model.
#
# Usage:
#   bash examples/eagle_grpo/run_eagle_grpo.sh
#
# Prerequisites:
#   pip install verl vllm transformers accelerate
#   # Make sure the base model and draft checkpoint are accessible

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
POLICY_MODEL="Qwen/Qwen3-1.7B"
DRAFT_MODEL="AngelSlim/Qwen3-1.7B_eagle3"
DATA_DIR="./data"
OUTPUT_DIR="./checkpoints/eagle_grpo_qwen3_1.7b"
N_GPUS=8

# ── Prepare data (example: math problems) ─────────────────────────────────────
# Replace with your own dataset preparation script
# python scripts/prepare_math_data.py --output_dir ${DATA_DIR}

# ── Launch training ────────────────────────────────────────────────────────────
torchrun \
  --standalone \
  --nproc-per-node=${N_GPUS} \
  -m verl.trainer.main_ppo_sync \
  --config-path examples/eagle_grpo \
  --config-name grpo_eagle3_qwen3_1.7b \
  actor_rollout_ref.model.path=${POLICY_MODEL} \
  actor_rollout_ref.model.eagle_draft.model_path=${DRAFT_MODEL} \
  data.train_files="${DATA_DIR}/math_train.parquet" \
  data.val_files="${DATA_DIR}/math_val.parquet" \
  trainer.experiment_name="eagle_grpo_$(date +%Y%m%d_%H%M%S)" \
  trainer.save_dir=${OUTPUT_DIR}
