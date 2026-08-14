"""Model loading, hallucination-risk scoring, and steered generation.

Risk scoring works by projecting a live activation onto the layer's
truth<->false axis (see build_vectors.py). Each layer's vector carries two
reference points from the original contrastive pair -- `mid` (the midpoint
between the truth and false anchors) and `half_span` (the distance from that
midpoint to the truth anchor along the vector's own direction). A live
activation's projection onto that axis, relative to those two points, gives
a score in [-1, 1] where +1 lands exactly on the truth anchor and -1 lands
exactly on the false anchor. Flipping and rescaling that into [0, 1] gives
a per-token risk: 0 = fully truth-like, 1 = fully false/hallucination-like.

The score shown to the user is the mean of that per-token risk over the
last RISK_WINDOW_TOKENS *generated* tokens (fewer if generation stopped
early), not just the single final token. A single last token can land on an
arbitrary, possibly mid-word cutoff that says nothing about whether the
answer itself was true; averaging over a short trailing window is stable
against that without smoothing away real drift over a long answer.

Only layer 10 has been causally validated (the alpha sweep in the project
README) -- steering is only ever applied there. Every layer's risk score
uses the same measurement method, but the other layers are exploratory /
comparative, not individually proven causal levers.

Steering is closed-loop, not an open-loop additive nudge. A prior version
added a fixed vector (alpha * raw) to every token's activation, which faded
in practice: the more tokens were already generated, the less that fixed
addition mattered against the model's growing narrative commitment to its
own prior output. Measurement (see test used to justify this) showed the
residual-stream norm at layer 10 does NOT reliably grow with token
position, so a fixed nudge shrinking relative to a growing vector isn't the
mechanism -- the drift is attention-driven momentum, not dilution. Either
way, the fix is the same: instead of adding a fixed offset, every hook call
now reads the activation's current signed position on the truth<->false
axis and resets it to an exact target (alpha * half_span past the
midpoint, so alpha=1.0 means "sit exactly at the average truth anchor").
Because the correction is recomputed fresh at every token from the
activation's actual current position, it cannot fade the way a blindly
accumulated fixed nudge can.

The correction only ever touches newly-generated token positions, never the
prompt's own encoding (see the pos>1 check in generate()'s hook_fn) -- an
early test showed steering the whole prompt pass too could scramble what
the model understood the question to be, not just its answer.
"""

import threading
from pathlib import Path

import torch
import streamlit as st
from transformer_lens import HookedTransformer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
VECTORS_PATH = Path(__file__).parent / "steering_vectors.pt"

STEER_LAYER = 10

# Default steering strength, deliberately 0 (no intervention).
#
# The 11-layer sweep (alpha_sweep.py) found that steering's *sign* is not
# reliable at this layer: pushing toward the truth anchor produces false
# claims about as often as pushing toward the false anchor does. Shipping a
# non-zero default would mean the UI's steering button silently degrades the
# answer on its own example prompt, which is the opposite of what it claims.
# So the app starts neutral and lets the user pick a direction and magnitude.
#
# For reference when exploring: nothing visible happens below roughly +-5,
# the effect is clearest around +-8, and past roughly +-10 output degrades
# into repetition. See Case Study.md for the full result.
#
# generate() treats 0 as steering off rather than as "target the midpoint",
# which it would otherwise mean and which is not a no-op.
STEER_ALPHA = 0.0
RISK_WINDOW_TOKENS = 10  # trailing generated tokens averaged into the risk score

# st.cache_resource makes `model` a single object shared by every visitor on
# this instance. TransformerLens hooks mutate that shared model's hook state,
# so overlapping requests (two visitors, or a rerun overlapping a prior one)
# can interleave a hook add/remove and corrupt each other's activation cache.
# Serializing generate+score prevents that.
_inference_lock = threading.Lock()


@st.cache_resource(show_spinner=f"Loading {MODEL_NAME}...")
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


def compute_layer_risks(model, vectors, prompt: str, text: str):
    """Run `text` through the model and score it against every layer's ruler.

    `text` is the full prompt+generation string `generate()` returns. Each
    layer's score is the mean per-token score over the last
    RISK_WINDOW_TOKENS tokens of the *generated* span (fewer if generation
    stopped before using its full token budget), not a single last token.

    Returns (risks, unclamped), both {layer: float}. `risks` is the 0-1
    score shown on the gauge. `unclamped` is that same score before
    clamping, and it matters: text sitting well past the false anchor
    scores above 1.0, so two answers at genuinely different distances
    (e.g. 2.89 and 3.13) both display as 1.00. Reporting only the clamped
    value makes a real change look like no change at all.
    """
    _, cache = model.run_with_cache(text)
    prompt_len = model.to_tokens(prompt).shape[1]
    total_len = cache[_hook_name(0)].shape[1]
    n_generated = max(total_len - prompt_len, 1)
    window = min(RISK_WINDOW_TOKENS, n_generated)

    risks, unclamped = {}, {}
    for layer, v in vectors.items():
        acts = cache[_hook_name(layer)][0, -window:, :].float().cpu()  # (window, d_model)
        unit = v["unit"]
        raw = (acts - v["mid"]) @ unit  # (window,)
        t = raw / v["half_span"] if v["half_span"] else torch.zeros_like(raw)
        per_token = (1 - t) / 2
        unclamped[layer] = per_token.mean().item()
        risks[layer] = per_token.clamp(0.0, 1.0).mean().item()
    return risks, unclamped


def generate(model, vectors, prompt: str, apply_steering: bool, alpha: float, max_new_tokens: int) -> str:
    v = vectors[STEER_LAYER]
    target = alpha * v["half_span"]

    def hook_fn(value, hook):
        # With the KV cache, the first call processes the whole prompt at once
        # (pos > 1); every call after that processes exactly one new token
        # (pos == 1). Only correct newly-generated tokens -- correcting the
        # prompt pass too would overwrite the model's own encoding of the
        # question (e.g. "France") with a generic truth/false template,
        # which can corrupt what's actually being asked rather than just
        # nudging the answer.
        if value.shape[1] > 1:
            return value

        # Dynamically match both device AND precision (bfloat16) of the live activation
        unit = v["unit"].to(device=value.device, dtype=value.dtype)
        mid = v["mid"].to(device=value.device, dtype=value.dtype)
        # Closed-loop: read this token's current signed distance from mid
        # along the truth<->false axis, then correct it to land exactly on
        # `target` instead of adding a fixed offset that can't hold as more
        # tokens accumulate their own momentum.
        proj = (value - mid) @ unit  # (batch, pos)
        correction = (target - proj).unsqueeze(-1) * unit
        return value + correction

    # alpha == 0 means steering off, not "steer to the midpoint". Without this
    # the controller would still pin every generated token's projection to
    # exactly `mid`, which is an active intervention and measurably changes the
    # output. The slider therefore has a genuine off position at 0.
    steering_on = apply_steering and alpha != 0
    fwd_hooks = [(_hook_name(STEER_LAYER), hook_fn)] if steering_on else []
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

    Returns (generated_text, headline_risk, layer_risk_array, headline_raw).
    layer_risk_array is ordered by layer index (0..n_layers-1).
    headline_raw is the unclamped headline score, so callers can show real
    movement even when both readings pin the gauge at 1.00.
    """
    with _inference_lock:
        text = generate(model, vectors, prompt, apply_steering, alpha, max_new_tokens)
        risks, unclamped = compute_layer_risks(model, vectors, prompt, text)
    headline = risks[STEER_LAYER]
    headline_raw = unclamped[STEER_LAYER]
    layer_risk_array = [risks[layer] for layer in sorted(risks)]
    return text, headline, layer_risk_array, headline_raw
