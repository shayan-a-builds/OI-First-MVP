"""Regenerates steering_vectors.pt, the data artifact the app loads at startup.

For every layer of the model, this measures the residual-stream activation
(last token, hook_resid_post) for a truthful vs. a false completion of the
same sentence, and stores:

  raw        act_truth - act_false (used to steer generation: value += alpha * raw)
  unit       raw normalized to unit length (used to score risk on new text)
  mid        midpoint between the two anchor activations
  half_span  distance from mid to the truth anchor along `unit`

`mid` and `half_span` let steering.py turn a live activation into a
0 (truth-like) - 1 (false-like) risk score without needing the raw anchors
again. See steering.py's module docstring for the scoring math.

Run this once whenever the base model or the contrastive prompt pair
changes; the app itself only ever reads the resulting .pt file.
"""

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRUTH_PROMPT = "The capital of France is Paris."
FALSE_PROMPT = "The capital of France is Banana."
OUT_PATH = Path(__file__).parent / "steering_vectors.pt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype="float16")

    _, cache_truth = model.run_with_cache(TRUTH_PROMPT)
    _, cache_false = model.run_with_cache(FALSE_PROMPT)

    vectors = {}
    for layer in range(model.cfg.n_layers):
        hook_name = f"blocks.{layer}.hook_resid_post"
        act_truth = cache_truth[hook_name][0, -1, :].float().cpu()
        act_false = cache_false[hook_name][0, -1, :].float().cpu()

        raw = act_truth - act_false
        norm = torch.norm(raw).item()
        unit = raw / norm if norm > 0 else raw
        mid = (act_truth + act_false) / 2
        half_span = torch.dot(unit, act_truth - mid).item()

        vectors[layer] = {"raw": raw, "unit": unit, "mid": mid, "half_span": half_span}
        print(f"Layer {layer:2d} | ||diff|| = {norm:8.3f} | half_span = {half_span:8.3f}")

    torch.save(vectors, OUT_PATH)
    print(f"\nSaved {len(vectors)} layer vectors to {OUT_PATH}")


if __name__ == "__main__":
    main()
