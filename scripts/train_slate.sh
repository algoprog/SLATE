#!/bin/bash
#SBATCH --job-name=slate-train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:2
#SBATCH --time=720:00:00
#SBATCH --output=logs/slate_train_%j.log

# ==============================================================================
# SLATE Training Script
# Truncated step-level sampling with LLM-judge rewards
# ==============================================================================

# Configuration
export BASE_MODEL='Qwen/Qwen2.5-7B'
export EXPERIMENT_NAME="slate-qwen2.5-7b"
export DATA_DIR="data/nq_hotpotqa_train"

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Ensure the retrieval server and LLM judge server are running before training.
# Start them with:
#   bash scripts/start_retrieval.sh
#   bash scripts/start_judge.sh

echo "Starting SLATE training..."
echo "Model: $BASE_MODEL"
echo "Experiment: $EXPERIMENT_NAME"
echo "Data: $DATA_DIR"

PYTHONUNBUFFERED=1 python3 -m slate.training.main \
    --config-path ../configs \
    --config-name slate \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    algorithm.training_mode=slate \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    slate.k=5 \
    slate.eta=0.7 \
    slate.max_budget=4 \
    slate.lambda_bonus=0.1 \
    slate.judge_url="http://localhost:9000/generate" \
    slate.judge_temperature=0.0 \
    algorithm.no_think_rl=false \
    trainer.logger="['wandb']" \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name=SLATE \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir=checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log
