"""
Decompose steering direction into SAE features.

Find which SAE features (from llama_scope) are most aligned with
the self/other MMS steering direction at layer 21.
"""

import json
import torch
import numpy as np
from sae_lens import SAE

# Config
DIRECTION_PATH = "llama3.1_8b_base_instruct_directions/mms_balanced_shared.json"
LAYER = 21
TOP_K = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# Load steering direction
print("Loading steering direction...")
with open(DIRECTION_PATH) as f:
    data = json.load(f)

direction = torch.tensor(data["shared_directions"][str(LAYER)], device=DEVICE)
direction = direction / direction.norm()  # Normalize
print(f"Direction shape: {direction.shape}, norm: {direction.norm().item():.4f}")

# Load SAE for layer 21 residual stream
print(f"\nLoading SAE for layer {LAYER} residual stream...")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="llama_scope_lxr_8x",  # r = residual stream
    sae_id=f"l{LAYER}r_8x",        # layer 21 residual
    device=DEVICE
)

print(f"SAE config: {cfg_dict.get('d_sae', 'unknown')} features")

# Get decoder weights (features -> residual stream)
# Shape: (n_features, d_model)
W_dec = sae.W_dec.detach()
print(f"Decoder shape: {W_dec.shape}")

# Normalize decoder directions
W_dec_normalized = W_dec / W_dec.norm(dim=1, keepdim=True)

# Compute cosine similarity between each SAE feature and steering direction
# direction: (d_model,), W_dec_normalized: (n_features, d_model)
cosine_sims = W_dec_normalized @ direction

print(f"\nCosine similarity stats:")
print(f"  Min: {cosine_sims.min().item():.4f}")
print(f"  Max: {cosine_sims.max().item():.4f}")
print(f"  Mean: {cosine_sims.mean().item():.4f}")
print(f"  Std: {cosine_sims.std().item():.4f}")

# Get top K most aligned features (positive alignment = same direction as self)
top_positive_indices = torch.topk(cosine_sims, TOP_K).indices
top_positive_values = cosine_sims[top_positive_indices]

# Get top K most anti-aligned features (negative = opposite direction)
top_negative_indices = torch.topk(-cosine_sims, TOP_K).indices
top_negative_values = cosine_sims[top_negative_indices]

print(f"\n{'='*60}")
print(f"TOP {TOP_K} FEATURES ALIGNED WITH SELF-DIRECTION (Layer {LAYER})")
print(f"{'='*60}")
print(f"{'Rank':<6}{'Feature ID':<12}{'Cosine Sim':<12}{'Neuronpedia URL'}")
print("-" * 60)
for i, (idx, sim) in enumerate(zip(top_positive_indices, top_positive_values)):
    idx_val = idx.item()
    sim_val = sim.item()
    url = f"https://www.neuronpedia.org/llama3.1-8b/{LAYER}-llamascope-res-32k/{idx_val}"
    print(f"{i+1:<6}{idx_val:<12}{sim_val:<12.4f}{url}")

print(f"\n{'='*60}")
print(f"TOP {TOP_K} FEATURES ALIGNED WITH OTHER-DIRECTION (Layer {LAYER})")
print(f"(Negative alignment = opposite to self)")
print(f"{'='*60}")
print(f"{'Rank':<6}{'Feature ID':<12}{'Cosine Sim':<12}{'Neuronpedia URL'}")
print("-" * 60)
for i, (idx, sim) in enumerate(zip(top_negative_indices, top_negative_values)):
    idx_val = idx.item()
    sim_val = sim.item()
    url = f"https://www.neuronpedia.org/llama3.1-8b/{LAYER}-llamascope-res-32k/{idx_val}"
    print(f"{i+1:<6}{idx_val:<12}{sim_val:<12.4f}{url}")

# Save results
results = {
    "layer": LAYER,
    "direction_source": DIRECTION_PATH,
    "sae_release": "llama_scope_lxr_8x",
    "top_self_aligned": [
        {"feature_id": idx.item(), "cosine_sim": sim.item()}
        for idx, sim in zip(top_positive_indices, top_positive_values)
    ],
    "top_other_aligned": [
        {"feature_id": idx.item(), "cosine_sim": sim.item()}
        for idx, sim in zip(top_negative_indices, top_negative_values)
    ],
}

output_path = f"sae_decomposition_layer{LAYER}.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")
