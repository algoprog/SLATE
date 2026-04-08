#!/bin/bash
#SBATCH --job-name=llm-judge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --time=720:00:00
#SBATCH --output=logs/judge_%j.log

# ==============================================================================
# Start LLM Judge Server for SLATE Training
# Uses vLLM to serve the judge model (Gemma3-27B or Qwen3-30B)
# ==============================================================================

# Set cache directories
export TRANSFORMERS_CACHE="../../hf_cache"
export HF_HOME="../../hf_cache"
export HF_DATASETS_CACHE="../../hf_cache"

# Judge model (change as needed)
JUDGE_MODEL="google/gemma-3-27b-it"
# Alternative: JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

PORT=9000

echo "Starting LLM judge server..."
echo "Model: $JUDGE_MODEL"
echo "Port: $PORT"

python3 -c "
import sys
sys.path.insert(0, '../..')
from llm.llm import LLMServer, app
import flask

server = LLMServer(model_name='${JUDGE_MODEL}')
app.run(host='0.0.0.0', port=${PORT}, debug=False)
"
