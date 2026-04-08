#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=256G  # Requested Memory
#SBATCH -p superpod-a100  # Partition
#SBATCH -t 48:00:00  # Wall Time
#SBATCH -G 1  # Number of GPUs
#SBATCH -o log-%j.out  # %j = job ID
#SBATCH -e log-%j.err  # %j = job ID
module add python/3.11.7
module add cuda/12.1
module add conda/latest

# Create and activate a virtual environment
conda activate searchr1

set -euo pipefail

corpus_file=/gypsum/work1/zamani/csamarinas/search-reasoner/Search-R1/index/wiki-18.jsonl # jsonl
save_dir=/gypsum/work1/zamani/csamarinas/search-reasoner/Search-R1/index
retriever_name=e5 # this is for indexing naming
retriever_model=intfloat/e5-base-v2
faiss_type="IVF65536_HNSW32,PQ128"
faiss_metric=ip

# Allow safe overrides without editing the script
max_length=${MAX_LENGTH:-256}
batch_size=${BATCH_SIZE:-256}
pooling_method=${POOLING_METHOD:-mean}

export HF_HOME=/gypsum/work1/zamani/csamarinas/hf_cache
export HF_DATASETS_CACHE=/gypsum/work1/zamani/csamarinas/hf_cache

# change retriever_name to bm25 for BM25 indexing
# adjust faiss_type/faiss_metric for alternative ANN layouts
cmd=(
  python3 index_builder.py
  --retrieval_method "$retriever_name"
  --model_path "$retriever_model"
  --corpus_path "$corpus_file"
  --save_dir "$save_dir"
  --use_fp16
  --max_length "$max_length"
  --batch_size "$batch_size"
  --pooling_method "$pooling_method"
  --faiss_type "$faiss_type"
  --faiss_metric "$faiss_metric"
  --save_embedding
)

echo "Building FAISS index (batch_size=${batch_size}, max_length=${max_length})"
"${cmd[@]}"
