# SLATE: Step-Level Advantage Estimation for Truncated Exploration

[![arxiv](https://img.shields.io/badge/arXiv-2602.23440-b31b1b.svg)](https://arxiv.org/abs/2602.23440)

SLATE is a training framework for retrieval-augmented LLM reasoning that addresses the credit assignment problem in multi-step search trajectories through two complementary innovations:

1. **Truncated Step-Level Sampling**: Instead of sampling k complete independent trajectories, SLATE generates k candidate actions from a shared prefix at each step, isolating variation to a single decision point. This achieves a provable T-fold reduction in advantage variance (Theorem 1).

2. **Dense, Decomposed Process Rewards**: An LLM judge separately evaluates reasoning quality, query quality, and answer correctness on a ternary scale {-1, 0, +1}, providing richer supervision than binary outcome signals.

This repository supports both SLATE training and standard Search-R1 GRPO training.

## Results

| Method | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | Musique | Bamboogle | Avg. |
|--------|-----|----------|-------|----------|-------|---------|-----------|------|
| Search-R1 (7B) | 0.480 | 0.638 | 0.457 | 0.433 | 0.382 | 0.196 | 0.432 | 0.431 |
| **SLATE (7B)** | **0.497** | **0.652** | **0.470** | **0.451** | **0.413** | **0.247** | **0.494** | **0.461** |

## Installation

```bash
# Clone the repository
cd SLATE

# Install dependencies
pip install -e ".[gpu,vllm]"

# Or install from requirements.txt
pip install -r requirements.txt
```

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.0
- CUDA-capable GPUs (A100 recommended)
- vLLM for model serving

## Quick Start

### 1. Start the Retrieval Server

The retrieval server provides search results during training and inference. You need a FAISS index and corpus file.

```bash
# Download data (NQ + HotpotQA training data)
# Place in data/nq_hotpotqa_train/

# Start retrieval server
bash scripts/start_retrieval.sh
```

### 2. Start the LLM Judge Server (SLATE mode only)

SLATE requires an LLM judge for decomposed reward evaluation during training.

```bash
bash scripts/start_judge.sh
```

### 3. Train

#### SLATE Training

```bash
bash scripts/train_slate.sh
```

Key hyperparameters (from paper):
- `k=5`: Number of candidate actions per step
- `eta=0.7`: Reward-weighted sampling temperature
- `lambda_bonus=0.1`: Early termination bonus
- `max_budget=4`: Maximum search steps
- LoRA rank 16, alpha 64
- Learning rate 1e-6
- 500 training steps on 2x A100

#### Standard Search-R1 Training

```bash
bash scripts/train_searchr1.sh
```

This replicates the original Search-R1 GRPO training with:
- `n_agent=5`: 5 full trajectory samples per prompt
- EM reward (binary outcome)
- 1005 training steps on 8x A100

### 4. Merge LoRA Adapter

After training, merge the LoRA adapter with the base model:

```bash
python merge_lora.py \
    --base_model Qwen/Qwen2.5-7B \
    --lora_adapter checkpoints/slate-qwen2.5-7b/global_step_500 \
    --output merged_models/slate-qwen2.5-7b
```

### 5. Evaluate

```bash
bash scripts/eval.sh merged_models/slate-qwen2.5-7b
```

## Project Structure

```
SLATE/
├── slate/                         # Core SLATE implementation
│   ├── agent_loops/
│   │   └── slate_agent_loop.py    # Truncated step-level sampling (Algorithm 1)
│   ├── rewards/
│   │   ├── llm_judge.py           # LLM-as-judge decomposed ternary rewards
│   │   └── em_reward.py           # Exact-match reward (Search-R1 baseline)
│   ├── training/
│   │   ├── main.py                # Training entry point (Hydra)
│   │   └── step_level_algos.py    # Step-level GRPO advantages + sampling
│   └── utils/
│       └── trajectory.py          # Trajectory parsing utilities
├── configs/
│   ├── slate.yaml                 # SLATE training config
│   └── searchr1.yaml              # Standard Search-R1 config
├── scripts/
│   ├── train_slate.sh             # SLATE training script
│   ├── train_searchr1.sh          # Search-R1 training script
│   ├── start_judge.sh             # Start LLM judge server
│   ├── start_retrieval.sh         # Start retrieval server
│   └── eval.sh                    # Evaluation script
├── eval/
│   ├── infer.py                   # Inference with search
│   └── evaluate.py                # Aggregate evaluation metrics
├── verl/                          # VERL framework (RL training infrastructure)
├── search_r1/                     # Search-R1 utilities (retrieval, tokenization)
├── merge_lora.py                  # Merge LoRA adapter with base model
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## Configuration

Training is configured via Hydra YAML files in `configs/`. Key configuration groups:

### SLATE-specific (`slate.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slate.k` | 5 | Number of candidate actions per step |
| `slate.eta` | 0.7 | Reward-weighted sampling temperature |
| `slate.max_budget` | 4 | Maximum action budget (search steps) |
| `slate.lambda_bonus` | 0.1 | Early-termination bonus coefficient |
| `slate.judge_url` | `http://localhost:9000/generate` | LLM judge server URL |
| `slate.judge_temperature` | 0.0 | Judge sampling temperature |

### Model (`actor_rollout_ref.model.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.path` | `Qwen/Qwen2.5-7B` | Base model |
| `model.lora_rank` | 16 | LoRA rank (0 to disable) |
| `model.lora_alpha` | 64 | LoRA scaling factor |

### Algorithm (`algorithm.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm.training_mode` | `slate` | Training mode: `slate` or `searchr1` |
| `algorithm.adv_estimator` | `grpo` | Advantage estimator |
| `algorithm.kl_ctrl.kl_coef` | 0.001 | KL regularization coefficient |

## How SLATE Works

### Training Loop (Algorithm 1)

For each question in the training batch:

1. **Initialize** prefix as empty, step t = 1
2. **While** t <= B (max budget):
   - Generate **k candidate actions** from the shared prefix (think + search/answer)
   - **Evaluate** each candidate with the LLM judge:
     - Search steps: r_t = r_think + r_query
     - Answer steps: r_t = r_think + r_answer + lambda * (B-t)/B
   - Compute **step-level GRPO advantages**: A_t^(j) = (r_t^(j) - mean) / (std + eps)
   - **Accumulate gradients** using clipped policy gradient with loss masking
   - **Select** winning action via reward-weighted sampling (temperature eta)
   - Extend prefix with selected action + retrieved documents
3. **Update** model parameters with accumulated gradients

### Reward Decomposition

The LLM judge evaluates each candidate on a ternary scale {-1, 0, +1}:

- **Thinking reward** (r_think): Evaluates relevance, clarity, specificity, progress, faithfulness
- **Query reward** (r_query): Evaluates relevance, specificity, searchability, alignment, novelty
- **Answer reward** (r_answer): Evaluates correctness against ground truth

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
