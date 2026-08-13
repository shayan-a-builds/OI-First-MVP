"""Model loading, hallucination-risk scoring, and steered generation.

Risk scoring works by projecting a live activation onto the layer's
truth<->false axis (see build_vectors.py). Each layer's vector carries two
reference points from the original contrastive pair -- `mid` (the midpoint
between the truth and false anchors) and `half_span` (the distance from that
midpoint to the truth anchor along the vector's own direction). A live
activation's projection onto that axis, relative to those two points, gives
a score in [-1, 1] where +1 lands exactly on the truth anchor and -1 lands
exactly on the false anchor. Flipping and rescaling that into [0, 1] gives
the risk score: 0 = fully truth-like, 1 = fully false/hallucination-like.

Only layer 10 has been causally validated (the alpha sweep in the project
README) -- steering is only ever applied there. Every layer's risk score
uses the same measurement method, but the other layers are exploratory /
comparative, not individually proven causal levers.
"""

from pathlib import Path

import torch
import streamlit as st
from transformer_lens import HookedTransformer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VECTORS_PATH = Path(__file__).parent / "steering_vectors.pt"

STEER_LAYER = 10
STEER_ALPHA = 1.0  # the "Goldilocks zone" identified by the alpha sweep


@st.cache_resource(show_spinner="Loading Qwen2.5-0.5B-Instruct...")
def load_model():
    device = "cpu"
    return HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, device=device, dtype="bfloat16", low_cpu_mem_usage=True
    )

@st.cache_resource(show_spinner="Loading steering vectors...")
def load_vectors():
    return torch.load(VECTORS_PATH, map_location="cpu")


def _hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


def compute_layer_risks(model, vectors, text: str) -> dict[int, float]:
    """Run `text` through the model and return {layer: risk in [0, 1]}."""
    _, cache = model.run_with_cache(text)
    device = model.cfg.device

    risks = {}
    for layer, v in vectors.items():
        act = cache[_hook_name(layer)][0, -1, :].float().cpu()
        unit = v["unit"]
        raw = torch.dot(unit, act - v["mid"]).item()
        t = raw / v["half_span"] if v["half_span"] else 0.0
        risks[layer] = min(max((1 - t) / 2, 0.0), 1.0)
    return risks


def generate(model, vectors, prompt: str, apply_steering: bool, alpha: float, max_new_tokens: int) -> str:
    device = model.cfg.device

    def hook_fn(value, hook):
    # Dynamically match both device AND precision (bfloat16) of the live activation
        vec = vectors[STEER_LAYER]["raw"].to(device=value.device, dtype=value.dtype)
        value[:, :, :] += alpha * vec
        return value

    fwd_hooks = [(_hook_name(STEER_LAYER), hook_fn)] if apply_steering else []
    with model.hooks(fwd_hooks=fwd_hooks):
        output = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            verbose=False,
        )
    return output.strip()


def run(model, vectors, prompt: str, apply_steering: bool, alpha: float = STEER_ALPHA, max_new_tokens: int = 40):
    """Generate (optionally steered) and score the result.

    Returns (generated_text, headline_risk, layer_risk_array) where
    layer_risk_array is ordered by layer index (0..n_layers-1).
    """
    text = generate(model, vectors, prompt, apply_steering, alpha, max_new_tokens)
    risks = compute_layer_risks(model, vectors, text)
    headline = risks[STEER_LAYER]
    layer_risk_array = [risks[layer] for layer in sorted(risks)]
    return text, headline, layer_risk_array
