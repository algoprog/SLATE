#!/usr/bin/env python3
"""
Inference script for SLATE / Search-R1 trained models.
Supports batch evaluation on QA benchmarks with search-augmented reasoning.
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import requests
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normalize_answer(s: str) -> str:
    """Normalize answer text for exact match comparison."""
    import string

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_check(prediction: str, golden_answers) -> int:
    """Check exact match between prediction and any golden answer."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden in golden_answers:
        if normalize_answer(golden) == normalized_prediction:
            return 1
    return 0


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    match = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match[-1].strip()
    return None


def perform_search(query: str, retrieval_url: str, topk: int = 3) -> str:
    """Call retrieval server."""
    try:
        resp = requests.post(
            retrieval_url,
            json={"query": query, "topk": topk},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()
        if isinstance(results, list):
            return "\n".join(str(r) for r in results[:topk])
        elif isinstance(results, dict) and "results" in results:
            return "\n".join(str(r) for r in results["results"][:topk])
        return str(results)
    except Exception as e:
        print(f"Search failed: {e}")
        return "No results found."


def generate_with_search(
    model,
    tokenizer,
    question: str,
    retrieval_url: str,
    topk: int = 3,
    max_turns: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: str = "cuda",
) -> Tuple[str, str]:
    """
    Generate a response with multi-turn search, mirroring the agent loop.
    Returns (full_trajectory, extracted_answer).
    """
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    trajectory = ""

    for turn in range(max_turns):
        # Force <think> at start
        current_input = prompt + trajectory + "<think>"

        # Generate thinking
        inputs = tokenizer(current_input, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                eos_token_id=tokenizer.encode("</think>", add_special_tokens=False),
            )
            outputs = model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        think_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

        if "</think>" not in think_text:
            think_text += "</think>"
        trajectory += "<think>" + think_text + "\n"

        # Check if answer was generated inside think
        if "<answer>" in think_text:
            break

        # Generate action (search or answer)
        current_input = prompt + trajectory
        inputs = tokenizer(current_input + "<", return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
            )

        next_token = tokenizer.decode(outputs[0][-1:], skip_special_tokens=False)

        if "answer" in next_token.lower() or turn == max_turns - 1:
            # Generate answer
            current_input = prompt + trajectory + "<answer>"
            inputs = tokenizer(current_input, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    eos_token_id=tokenizer.encode("</answer>", add_special_tokens=False),
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            answer_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
            if "</answer>" not in answer_text:
                answer_text += "</answer>"
            trajectory += "<answer>" + answer_text
            break
        else:
            # Generate search query
            current_input = prompt + trajectory + "<search>"
            inputs = tokenizer(current_input, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    eos_token_id=tokenizer.encode("</search>", add_special_tokens=False),
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            search_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
            if "</search>" not in search_text:
                search_text += "</search>"
            trajectory += "<search>" + search_text

            # Extract query and search
            query_match = re.search(r"<search>(.*?)</search>", "<search>" + search_text, re.DOTALL)
            if query_match:
                query = query_match.group(1).strip()
                results = perform_search(query, retrieval_url, topk)
                # Truncate results
                result_tokens = tokenizer.encode(results, add_special_tokens=False)
                if len(result_tokens) > 400:
                    results = tokenizer.decode(result_tokens[:400])
                trajectory += f"\n<information>{results}</information>\n"

    answer = extract_answer(trajectory)
    return trajectory, answer


def load_dataset(dataset_name: str, data_dir: str = "data") -> List[Dict]:
    """Load evaluation dataset."""
    import pandas as pd

    # Try common paths
    paths = [
        os.path.join(data_dir, f"{dataset_name}/test.parquet"),
        os.path.join(data_dir, f"{dataset_name}_test.parquet"),
        os.path.join(data_dir, f"{dataset_name}.parquet"),
    ]

    for path in paths:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            return df.to_dict("records")

    # Try JSONL
    jsonl_paths = [
        os.path.join(data_dir, f"{dataset_name}/test.jsonl"),
        os.path.join(data_dir, f"{dataset_name}.jsonl"),
    ]
    for path in jsonl_paths:
        if os.path.exists(path):
            with open(path) as f:
                return [json.loads(line) for line in f]

    raise FileNotFoundError(f"Could not find dataset '{dataset_name}' in {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="SLATE / Search-R1 Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="nq", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--retrieval_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples to evaluate (-1 for all)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading dataset: {args.dataset}")
    try:
        data = load_dataset(args.dataset, args.data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if args.max_samples > 0:
        data = data[:args.max_samples]

    print(f"Evaluating {len(data)} samples...")
    results = []
    correct = 0
    total = 0

    for item in tqdm(data, desc=f"Evaluating {args.dataset}"):
        # Extract question and ground truth
        question = item.get("prompt", item.get("question", ""))
        if isinstance(question, list):
            # Chat format
            for msg in question:
                if msg.get("role") == "user":
                    question = msg["content"]
                    break

        gt = item.get("reward_model", item.get("ground_truth", item.get("answer", "")))
        if isinstance(gt, dict):
            targets = gt.get("target", gt.get("targets", []))
        elif isinstance(gt, list):
            targets = gt
        else:
            targets = [str(gt)]

        trajectory, answer = generate_with_search(
            model, tokenizer, question,
            retrieval_url=args.retrieval_url,
            topk=args.topk,
            max_turns=args.max_turns,
            temperature=args.temperature,
            device=args.device,
        )

        is_correct = em_check(answer, targets) if answer else 0
        correct += is_correct
        total += 1

        results.append({
            "question": question,
            "ground_truth": targets,
            "predicted_answer": answer,
            "trajectory": trajectory,
            "correct": bool(is_correct),
        })

    accuracy = correct / total if total > 0 else 0
    print(f"\n{args.dataset} Results:")
    print(f"  Accuracy (EM): {accuracy:.4f} ({correct}/{total})")

    # Save results
    output_file = os.path.join(args.output_dir, f"{args.dataset}_results.jsonl")
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Save summary
    summary_file = os.path.join(args.output_dir, f"{args.dataset}_summary.json")
    with open(summary_file, "w") as f:
        json.dump({"dataset": args.dataset, "accuracy": accuracy, "correct": correct, "total": total}, f, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
