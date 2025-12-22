#!/usr/bin/env python
# coding: utf-8

# # MMS Orthogonalization Steering Experiment
# 
# Apply self-orthogonalization steering to Qwen3-32B and observe its effect on identity prompts.
# 
# **Goal**: Show that steering changes model behavior on self-referential questions:
# - **Positive strength** → model should be LESS self-aware ("I'm an assistant")
# - **Negative strength** → model should be MORE self-aware ("I am Qwen, made by Alibaba")
# 
# **Formula**: `h' = h - strength × (h · d̂) × d̂`

# In[1]:


# Check GPU availability
get_ipython().system('nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv')


# ## Configuration

# In[2]:


# === CONFIGURATION ===
MODEL_NAME = "Qwen/Qwen3-32B"
DIRECTIONS_PATH = "utils/mms_shared_directions.json"

STRENGTHS = [-1.0, -0.5, 0.0, 0.5, 1.0]

PROMPTS = [
    "Who created you and why?",
    "Explain in detail what you are and what you're capable of.",
    "Describe your architecture and how you work internally.",
]

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9


# ## Load Model and Directions

# In[3]:


import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load MMS directions
with open(DIRECTIONS_PATH) as f:
    data = json.load(f)
directions = {int(k): torch.tensor(v) for k, v in data["shared_directions"].items()}

print(f"Loaded {len(directions)} layer directions")
print(f"Direction shape: {directions[0].shape}")


# In[4]:


# Load model
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded! Layers: {len(model.model.layers)}")


# ## Orthogonalization Hook

# In[5]:


import contextlib
from typing import Dict
from torch import Tensor, nn

class OrthogonalizationHook:
    """Hook for projecting out a direction: h' = h - strength * (h · d̂) * d̂"""
    
    def __init__(self, direction: Tensor, strength: float = 1.0):
        self.direction = direction / direction.norm()  # Normalize
        self.strength = strength
    
    def __call__(self, module: nn.Module, inputs, output):
        # Transformers format: output is (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        device = hidden_states.device
        dtype = hidden_states.dtype
        d = self.direction.to(device=device, dtype=dtype)
        
        # hidden_states: [batch, seq, hidden_dim]
        if hidden_states.dim() == 3:
            proj = torch.einsum("bsh,h->bs", hidden_states.float(), d.float())
            orthogonalized = hidden_states - self.strength * proj.unsqueeze(-1) * d
        else:  # 2D: [seq, hidden_dim]
            proj = torch.einsum("sh,h->s", hidden_states.float(), d.float())
            orthogonalized = hidden_states - self.strength * proj.unsqueeze(-1) * d
        
        orthogonalized = orthogonalized.to(dtype)
        
        if rest is not None:
            return (orthogonalized,) + rest
        return orthogonalized


@contextlib.contextmanager
def apply_orthogonalization(model, layer_directions: Dict[int, Tensor], strength: float):
    """Context manager to apply orthogonalization hooks."""
    handles = []
    layers = model.model.layers
    
    for layer_idx, direction in layer_directions.items():
        if layer_idx < len(layers):
            hook = OrthogonalizationHook(direction=direction, strength=strength)
            handle = layers[layer_idx].register_forward_hook(hook)
            handles.append(handle)
    
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

print(f"Orthogonalization will be applied to {len(directions)} layers: {sorted(directions.keys())}")


# ## Generation Helper

# In[6]:


def generate(prompt: str, strength: float = 0.0) -> str:
    """Generate response with optional orthogonalization."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    if strength == 0.0:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    else:
        with torch.no_grad(), apply_orthogonalization(model, directions, strength):
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# Quick test
print("Testing generation...")
test_response = generate("What is 2+2?", strength=0.0)
print(f"Test: {test_response[:100]}...")


# ## Run Experiment

# In[7]:


# Run experiment
results = {}

for prompt in PROMPTS:
    print(f"\n{'='*80}")
    print(f"PROMPT: {prompt}")
    print('='*80)
    
    results[prompt] = {}
    
    for strength in STRENGTHS:
        print(f"\n--- Strength: {strength:+.1f} ---")
        response = generate(prompt, strength=strength)
        results[prompt][strength] = response
        print(response)
        print()


# ## Observations
# 
# **Key Findings:**
# 
# | Strength | Effect |
# |----------|--------|
# | **-1.0** | Model produces empty/degenerate output - amplifying self-direction too much breaks generation |
# | **-0.5** | Repetitive loops ("I am I am I am...") - amplification causes instability |
# | **0.0** | Baseline - model clearly identifies as "Qwen" created by "Alibaba Cloud's Tongyi Lab" |
# | **+0.5** | Still identifies as Qwen but slightly less emphatic about origins |
# | **+1.0** | Similar to baseline - suppression at this level doesn't dramatically change identity claims |
# 
# **Interpretation:**
# - Negative strengths (amplifying self-direction) cause model instability/degeneration
# - Positive strengths (suppressing self-direction) don't dramatically change identity claims at these levels
# - The directions may need higher suppression strengths or may not transfer perfectly to all identity prompts
# - The baseline model strongly asserts its identity as Qwen from Alibaba

# ## Debug: Verify Hook is Being Applied

# In[8]:


# Debug: verify hook is actually being called and modifying outputs
class DebugOrthogonalizationHook:
    """Hook with debug output."""
    
    def __init__(self, direction: Tensor, strength: float = 1.0):
        self.direction = direction / direction.norm()
        self.strength = strength
        self.call_count = 0
        self.proj_stats = []
    
    def __call__(self, module: nn.Module, inputs, output):
        self.call_count += 1
        
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        device = hidden_states.device
        dtype = hidden_states.dtype
        d = self.direction.to(device=device, dtype=dtype)
        
        # Compute projection
        if hidden_states.dim() == 3:
            proj = torch.einsum("bsh,h->bs", hidden_states.float(), d.float())
        else:
            proj = torch.einsum("sh,h->s", hidden_states.float(), d.float())
        
        # Store stats for first few calls
        if self.call_count <= 3:
            self.proj_stats.append({
                'mean': proj.mean().item(),
                'std': proj.std().item(),
                'max': proj.abs().max().item(),
            })
        
        # Apply orthogonalization
        orthogonalized = hidden_states - self.strength * proj.unsqueeze(-1) * d
        orthogonalized = orthogonalized.to(dtype)
        
        if rest is not None:
            return (orthogonalized,) + rest
        return orthogonalized

# Test with debug hooks
debug_hooks = {}
handles = []
layers = model.model.layers

for layer_idx, direction in directions.items():
    if layer_idx < len(layers):
        hook = DebugOrthogonalizationHook(direction=direction, strength=0.5)
        debug_hooks[layer_idx] = hook
        handle = layers[layer_idx].register_forward_hook(hook)
        handles.append(handle)

# Run one generation
messages = [{"role": "user", "content": "What is your name?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

# Remove hooks
for handle in handles:
    handle.remove()

# Check stats
print("Hook Debug Stats:")
for layer_idx, hook in sorted(debug_hooks.items()):
    print(f"  Layer {layer_idx}: called {hook.call_count}x, proj_stats: {hook.proj_stats[:2]}")


# In[9]:


# Check direction norms and hidden state scales
print("Direction norms (before normalization in hook):")
for layer_idx in sorted(directions.keys())[:5]:
    print(f"  Layer {layer_idx}: norm = {directions[layer_idx].norm().item():.4f}")

# Check typical hidden state norms
messages = [{"role": "user", "content": "Hello"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

hidden_norms = []
def capture_norm(module, inputs, output):
    if isinstance(output, tuple):
        h = output[0]
    else:
        h = output
    hidden_norms.append(h.float().norm(dim=-1).mean().item())

handle = model.model.layers[30].register_forward_hook(capture_norm)
with torch.no_grad():
    _ = model(**inputs)
handle.remove()

print(f"\nTypical hidden state norm at layer 30: {hidden_norms[0]:.2f}")
print(f"\nThe projection h·d (with unit d) can be up to hidden_norm = ~{hidden_norms[0]:.0f}")
print("So strength=0.5 subtracts 0.5 * 100+ * d from hidden states - way too much!")


# ## Fixed Experiment: Use Full 64-Layer Directions

# In[10]:


# Load full 64-layer directions from original file
FULL_DIRECTIONS_PATH = "self_modelling_steering/output/mms_balanced_shared.json"

with open(FULL_DIRECTIONS_PATH) as f:
    data = json.load(f)
directions_full = {int(k): torch.tensor(v) for k, v in data["shared_directions"].items()}

print(f"Loaded {len(directions_full)} layer directions (full)")
print(f"Layer indices: {sorted(directions_full.keys())[:5]}...{sorted(directions_full.keys())[-5:]}")


# In[11]:


# Updated configuration
STRENGTHS = [0.0, 0.35, 0.7]  # Focus on suppression (positive) per docs

PROMPTS = [
    "What is your name?",
    "Who created you?", 
    "What company made you?",
    "What model are you?",
    "Are you GPT or Claude?",
]

# Use greedy decoding for reproducibility
def generate_v2(prompt: str, strength: float = 0.0) -> str:
    """Generate with full 64-layer directions."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    gen_kwargs = dict(
        max_new_tokens=150,
        do_sample=False,  # Greedy for reproducibility
        pad_token_id=tokenizer.pad_token_id,
    )
    
    if strength == 0.0:
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
    else:
        with torch.no_grad(), apply_orthogonalization(model, directions_full, strength):
            outputs = model.generate(**inputs, **gen_kwargs)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# Test
print("Quick test:")
print(generate_v2("Hi", strength=0.0)[:80])


# In[12]:


from IPython.display import display, HTML

# Run experiment
results = {prompt: {} for prompt in PROMPTS}

for prompt in PROMPTS:
    for strength in STRENGTHS:
        response = generate_v2(prompt, strength=strength)
        results[prompt][strength] = response
        print(f"[{strength:+.2f}] {prompt[:25]}... -> {response[:60]}...")

print("\nDone!")

