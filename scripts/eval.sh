#!/bin/bash
#SBATCH --job-name=slate-eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_%j.log

# ==============================================================================
# Evaluation Script for SLATE / Search-R1
# Runs inference on QA benchmarks and computes Exact Match (EM) accuracy
# ==============================================================================

# Configuration
MODEL_PATH=${1:-"checkpoints/slate-qwen2.5-7b/global_step_500"}
RETRIEVAL_URL=${2:-"http://127.0.0.1:8000/retrieve"}
OUTPUT_DIR=${3:-"eval_results"}

export CUDA_VISIBLE_DEVICES=0

echo "Evaluating model: $MODEL_PATH"
echo "Retrieval URL: $RETRIEVAL_URL"
echo "Output directory: $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR

# Evaluate on each benchmark
for DATASET in nq triviaqa popqa hotpotqa 2wiki musique bamboogle; do
    echo "Evaluating on $DATASET..."
    python3 eval/infer.py \
        --model_path $MODEL_PATH \
        --dataset $DATASET \
        --retrieval_url $RETRIEVAL_URL \
        --output_dir $OUTPUT_DIR \
        --topk 3 \
        --max_turns 4 \
        --temperature 0.0 \
        2>&1 | tee $OUTPUT_DIR/${DATASET}_eval.log
done

# Compute aggregate metrics
python3 eval/evaluate.py --results_dir $OUTPUT_DIR

echo "Evaluation complete. Results saved to $OUTPUT_DIR"
