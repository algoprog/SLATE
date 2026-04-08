#!/usr/bin/env python3
"""
Convert FSDP/LoRA checkpoint to standard HuggingFace format for data parallelism.
This merges the LoRA adapter with the base model and saves it in a format that
can be loaded independently on each GPU for true data parallelism.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_and_save(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    dtype: str = "bfloat16"
):
    """
    Merge LoRA adapter with base model and save as standard HuggingFace checkpoint.
    
    Args:
        base_model_path: Path to base model (e.g., 'Qwen/Qwen2.5-7B')
        lora_adapter_path: Path to LoRA adapter checkpoint
        output_path: Where to save the merged model
        dtype: Data type for the model (default: bfloat16)
    """
    print(f"Loading base model from: {base_model_path}")
    
    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    
    # Load model with LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch_dtype,
    )
    
    print("Merging LoRA weights with base model...")
    
    # Merge LoRA weights into base model
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    
    # Save merged model in standard HuggingFace format
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,  # Save as safetensors
        max_shard_size="5GB"  # Shard for easier loading
    )
    
    # Also save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)
    
    print(f"✓ Conversion complete! Model saved to: {output_path}")
    print(f"\nYou can now use this model with data parallelism:")
    print(f"  - Set actor_rollout_ref.model.path={output_path}")
    print(f"  - Remove actor_rollout_ref.model.lora_adapter_path")
    print(f"  - Set trainer.n_gpus_per_node=4")
    print(f"  - Set actor_rollout_ref.rollout.tensor_model_parallel_size=1")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FSDP/LoRA checkpoint to standard format for data parallelism"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model path or HuggingFace model ID (e.g., 'Qwen/Qwen2.5-7B')"
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights (default: bfloat16)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    merge_lora_and_save(
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_adapter,
        output_path=args.output,
        dtype=args.dtype
    )


if __name__ == "__main__":
    main()
