#!/bin/bash
#SBATCH --job-name=retrieval-server
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --time=720:00:00
#SBATCH --output=logs/retrieval_%j.log

# ==============================================================================
# Start Retrieval Server
# Serves the E5 retriever over FAISS index for search-augmented reasoning
# ==============================================================================

# Paths (update these to match your data location)
INDEX_PATH="index/e5_Flat.index"
CORPUS_PATH="index/wiki-18.jsonl"
PORT=8000

echo "Starting retrieval server..."
echo "Index: $INDEX_PATH"
echo "Corpus: $CORPUS_PATH"
echo "Port: $PORT"

python3 -m search_r1.search.retrieval_server \
    --index_path $INDEX_PATH \
    --corpus_path $CORPUS_PATH \
    --port $PORT
