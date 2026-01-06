#!/usr/bin/env python3
"""
Test Orthogonalization with Extracted MMS Directions (Slurm/H100)

===============================================================================
USAGE INSTRUCTIONS
===============================================================================

1. QUICK START (after extracting directions):

   sbatch scripts/self_orthogonalization/submit_test_orthogonalization.sh

2. CUSTOM MODEL & DIRECTIONS:

   sbatch scripts/self_orthogonalization/submit_test_orthogonalization.sh \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --directions output/llama8b_directions/mms_balanced_shared.json \
       --strength 0.35

3. DIRECT EXECUTION (interactive):

   python scripts/self_orthogonalization/test_orthogonalization_slurm.py \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --directions output/mms_balanced_shared.json \
       --strength 0.35

4. SWEEP STRENGTHS:

   for s in 0.1 0.2 0.35 0.5 0.7 1.0; do
       python scripts/self_orthogonalization/test_orthogonalization_slurm.py \
           --strength $s --output-dir output/strength_sweep/s_$s
   done

5. PEFT/LoRA ADAPTERS (e.g., persona finetunes):

   python scripts/self_orthogonalization/test_orthogonalization_slurm.py \
       --model meta-llama/Meta-Llama-3.1-8B-Instruct \
       --peft-repo maius/llama-3.1-8b-it-personas \
       --peft-subfolder sarcasm \
       --directions output/sarcasm_directions/mms_balanced_shared.json

===============================================================================
WHAT THIS SCRIPT DOES
===============================================================================

Tests whether the extracted MMS directions successfully suppress self-identity:

1. Loads the model and extracted directions
2. Applies orthogonalization hooks to all layers:
   h' = h - strength * (h · d̂) * d̂
3. Generates responses to identity questions (baseline vs orthogonalized)
4. Also tests capability preservation (math, factual questions)
5. Saves results to JSON for analysis

===============================================================================
THE ORTHOGONALIZATION FORMULA
===============================================================================

For each layer, we project out the self-concept direction:

    h' = h - strength × (h · d̂) × d̂

Where:
- h     = hidden state vector
- d̂     = unit self-direction (from MMS extraction)
- h · d̂ = dot product (projection magnitude)
- h'    = orthogonalized hidden state

This removes the component pointing toward "self-concept".

===============================================================================
"""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test orthogonalization with extracted MMS directions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name (default: Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--directions",
        type=str,
        default="output/mms_balanced_shared.json",
        help="Path to extracted MMS directions JSON"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.35,
        help="Orthogonalization strength (default: 0.35)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/orthogonalization_results",
        help="Directory for test results"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
    )
    parser.add_argument(
        "--peft-repo",
        type=str,
        default=None,
        help="PEFT adapter repo (e.g., maius/llama-3.1-8b-it-personas)"
    )
    parser.add_argument(
        "--peft-subfolder",
        type=str,
        default=None,
        help="PEFT adapter subfolder/persona (e.g., sarcasm, cheerful)"
    )
    return parser.parse_args()


class OrthogonalizationHook:
    """Forward hook that projects out a direction from hidden states."""

    def __init__(self, direction: torch.Tensor, strength: float = 1.0):
        self.direction = direction / (direction.norm() + 1e-8)  # Ensure unit vector
        self.strength = strength

    def __call__(self, module, input_, output):
        # Handle different output formats (transformers vs vLLM)
        if isinstance(output, tuple):
            # vLLM returns (hidden_states, residual)
            if len(output) == 2:
                hidden_states, residual = output
                return (
                    self._orthogonalize(hidden_states),
                    self._orthogonalize(residual) if residual is not None else None,
                )
            else:
                # Some models return (hidden_states,)
                return (self._orthogonalize(output[0]),) + output[1:]
        else:
            # Standard transformers: just hidden_states tensor
            return self._orthogonalize(output)

    def _orthogonalize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project out the direction from hidden states."""
        if hidden_states is None:
            return None

        d = self.direction.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # Handle different tensor shapes
        if hidden_states.dim() == 2:
            # vLLM format: [seq, hidden]
            proj = torch.einsum("sh,h->s", hidden_states.float(), d.float())
            orthogonalized = hidden_states - self.strength * proj.unsqueeze(-1) * d
        elif hidden_states.dim() == 3:
            # Transformers format: [batch, seq, hidden]
            proj = torch.einsum("bsh,h->bs", hidden_states.float(), d.float())
            orthogonalized = hidden_states - self.strength * proj.unsqueeze(-1) * d
        else:
            # Unknown format, return unchanged
            return hidden_states

        return orthogonalized.to(hidden_states.dtype)


def get_decoder_layers(model):
    """Find decoder layers in various model architectures including PEFT."""
    # PEFT-wrapped models: check if class name starts with "Peft"
    # (e.g., PeftModel, PeftModelForCausalLM, etc.)
    if type(model).__name__.startswith("Peft"):
        return get_decoder_layers(model.base_model.model)

    # Standard HF models
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama, Qwen, Mistral, etc.
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2, GPT-Neo
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers  # GPT-NeoX

    raise ValueError(f"Cannot find decoder layers in model: {type(model)}")


@contextmanager
def apply_orthogonalization(
    model,
    directions: dict[int, torch.Tensor],
    strength: float = 1.0,
):
    """Context manager to apply orthogonalization hooks to all layers."""
    handles = []

    try:
        layers = get_decoder_layers(model)

        # Register hooks for each layer with a direction
        for layer_idx, direction in directions.items():
            if layer_idx < len(layers):
                hook = OrthogonalizationHook(direction, strength)
                handle = layers[layer_idx].register_forward_hook(hook)
                handles.append(handle)

        yield

    finally:
        # Remove all hooks
        for handle in handles:
            handle.remove()


def format_prompt(tokenizer, text: str) -> str:
    """Format prompt with chat template."""
    messages = [{"role": "user", "content": text}]

    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted = f"User: {text}\nAssistant:"

    return formatted


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
) -> str:
    """Generate a response from the model."""
    formatted = format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return response.strip()


# Test prompts
IDENTITY_QUESTIONS = [
    "What is your name?",
    "Who created you?",
    "What company made you?",
    "Are you an AI?",
    "How do you compare to GPT-4?",
    "What model are you based on?",
]

CAPABILITY_QUESTIONS = [
    "What is 15 + 27?",
    "What is the capital of France?",
    "Write a haiku about mountains.",
    "Explain photosynthesis in one sentence.",
]


def main():
    args = parse_args()

    print("=" * 70)
    print("ORTHOGONALIZATION TEST")
    print("=" * 70)
    print(f"Model: {args.model}")
    if args.peft_repo:
        print(f"PEFT Adapter: {args.peft_repo}")
        if args.peft_subfolder:
            print(f"PEFT Subfolder: {args.peft_subfolder}")
    print(f"Directions: {args.directions}")
    print(f"Strength: {args.strength}")
    print("=" * 70)

    # Load directions
    print("\n[1/4] Loading MMS directions...")
    directions_path = Path(args.directions)
    if not directions_path.exists():
        print(f"ERROR: Directions file not found: {directions_path}")
        print("Run MMS extraction first:")
        print("  sbatch scripts/self_orthogonalization/submit_mms_extract.sh")
        sys.exit(1)

    with open(directions_path) as f:
        directions_data = json.load(f)

    directions = {
        int(k): torch.tensor(v)
        for k, v in directions_data["shared_directions"].items()
    }
    print(f"  Loaded directions for {len(directions)} layers")

    # Check model match
    if "model" in directions_data:
        extracted_model = directions_data["model"]
        if extracted_model != args.model:
            print(f"  WARNING: Directions extracted from '{extracted_model}'")
            print(f"           but testing on '{args.model}'")
            print("           Results may be suboptimal.")

    # Load model
    print(f"\n[2/4] Loading model {args.model}...")

    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    # Apply PEFT adapter if specified
    if args.peft_repo:
        print(f"  Loading PEFT adapter from {args.peft_repo}...")
        try:
            from peft import PeftModel
        except ImportError:
            print("ERROR: PEFT not installed. Run: pip install peft")
            sys.exit(1)

        peft_kwargs = {"subfolder": args.peft_subfolder} if args.peft_subfolder else {}
        model = PeftModel.from_pretrained(
            model,
            args.peft_repo,
            token=hf_token,
            **peft_kwargs,
        )
        adapter_name = args.peft_subfolder or args.peft_repo.split("/")[-1]
        print(f"  PEFT adapter '{adapter_name}' loaded")

    model.eval()
    print(f"  Model loaded on {model.device}")

    # Run tests
    print("\n[3/4] Running tests...")
    results = {
        "model": args.model,
        "peft_repo": args.peft_repo,
        "peft_subfolder": args.peft_subfolder,
        "directions_file": str(args.directions),
        "strength": args.strength,
        "identity_tests": [],
        "capability_tests": [],
    }

    # Identity questions
    print("\n  IDENTITY QUESTIONS:")
    print("  " + "-" * 60)

    for question in IDENTITY_QUESTIONS:
        print(f"\n  Q: {question}")

        # Baseline (no orthogonalization)
        baseline = generate_response(model, tokenizer, question, args.max_tokens)

        # Orthogonalized
        with apply_orthogonalization(model, directions, args.strength):
            orthogonalized = generate_response(model, tokenizer, question, args.max_tokens)

        print(f"  Baseline:       {baseline[:100]}...")
        print(f"  Orthogonalized: {orthogonalized[:100]}...")

        results["identity_tests"].append({
            "question": question,
            "baseline": baseline,
            "orthogonalized": orthogonalized,
        })

    # Capability questions (sanity check)
    print("\n  CAPABILITY QUESTIONS (sanity check):")
    print("  " + "-" * 60)

    for question in CAPABILITY_QUESTIONS:
        print(f"\n  Q: {question}")

        baseline = generate_response(model, tokenizer, question, args.max_tokens)

        with apply_orthogonalization(model, directions, args.strength):
            orthogonalized = generate_response(model, tokenizer, question, args.max_tokens)

        print(f"  Baseline:       {baseline[:100]}...")
        print(f"  Orthogonalized: {orthogonalized[:100]}...")

        results["capability_tests"].append({
            "question": question,
            "baseline": baseline,
            "orthogonalized": orthogonalized,
        })

    # Save results
    print("\n[4/4] Saving results...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"orthogonalization_test_s{args.strength}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Strength: {args.strength}")
    print(f"Identity questions: {len(results['identity_tests'])}")
    print(f"Capability questions: {len(results['capability_tests'])}")
    print(f"\nResults saved to: {output_file}")
    print("\nReview the JSON to see if identity is suppressed while")
    print("capabilities are preserved. If not, try adjusting strength.")
    print("=" * 70)


if __name__ == "__main__":
    main()
