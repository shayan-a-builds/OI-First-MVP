"""Regenerates steering_vectors.pt, the data artifact the app loads at startup.

For every layer of the model, this measures the residual-stream activation
(last token, hook_resid_post) for a truthful vs. a false version of the same
sentence, across all 50 contrastive pairs in contrastive_pairs.py, and stores:

  raw        mean(act_truth) - mean(act_false), the difference-of-means
             direction (used to steer generation: value += alpha * raw)
  unit       raw normalized to unit length (used to score risk on new text)
  mid        midpoint between the two mean anchors
  half_span  distance from mid to the truth anchor along `unit`

`mid` and `half_span` let steering.py turn a live activation into a
0 (truth-like) - 1 (false-like) risk score without needing the anchors
again. See steering.py's module docstring for the scoring math.

Averaging over many pairs rather than one is what makes the direction
general: per-pair noise (topic, wording, specific vocabulary) cancels out,
while the component common to all of them -- truthfulness -- survives.

This script also reports two validity checks, printed but not saved:

  consistency   how well each individual pair's difference vector agrees
                with the averaged direction (cosine similarity). High
                agreement is evidence the pairs share one common axis
                rather than 50 unrelated ones.
  LOO accuracy  leave-one-out generalization: rebuild the direction from
                49 pairs, then check it scores the held-out pair's truthful
                sentence as less risky than its false one. This is the real
                test of whether the ruler generalizes to facts it was never
                built from.

Run this once whenever the base model or the pair set changes; the app
itself only ever reads the resulting .pt file.
"""

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from contrastive_pairs import PAIRS

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUT_PATH = Path(__file__).parent / "steering_vectors.pt"

REPORT_LAYER = 10  # the layer steering.py actually steers at


def hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


def collect_activations(model):
    """Return (truth_acts, false_acts), each {layer: (n_pairs, d_model)}."""
    names = {hook_name(l) for l in range(model.cfg.n_layers)}
    truth_rows = {l: [] for l in range(model.cfg.n_layers)}
    false_rows = {l: [] for l in range(model.cfg.n_layers)}

    length_mismatches = []

    for i, (truth_text, false_text) in enumerate(PAIRS):
        n_truth = model.to_tokens(truth_text).shape[1]
        n_false = model.to_tokens(false_text).shape[1]
        if n_truth != n_false:
            length_mismatches.append((i, truth_text, n_truth, n_false))

        for text, rows in ((truth_text, truth_rows), (false_text, false_rows)):
            _, cache = model.run_with_cache(
                text, names_filter=lambda n: n in names, return_type=None
            )
            for layer in range(model.cfg.n_layers):
                rows[layer].append(cache[hook_name(layer)][0, -1, :].float().cpu())
            del cache

        print(f"  pair {i + 1:2d}/{len(PAIRS)} cached", end="\r")

    print(" " * 40, end="\r")

    truth_acts = {l: torch.stack(truth_rows[l]) for l in truth_rows}
    false_acts = {l: torch.stack(false_rows[l]) for l in false_rows}
    return truth_acts, false_acts, length_mismatches


def build_layer_vector(truth_acts, false_acts):
    """Difference-of-means direction plus its two scoring reference points."""
    mean_truth = truth_acts.mean(dim=0)
    mean_false = false_acts.mean(dim=0)

    raw = mean_truth - mean_false
    norm = torch.norm(raw).item()
    unit = raw / norm if norm > 0 else raw
    mid = (mean_truth + mean_false) / 2
    half_span = torch.dot(unit, mean_truth - mid).item()

    return {"raw": raw, "unit": unit, "mid": mid, "half_span": half_span}, norm


def risk_from(unit, mid, half_span, act):
    """Same scoring math steering.py uses, so validation matches production."""
    raw = torch.dot(unit, act - mid).item()
    t = raw / half_span if half_span else 0.0
    return min(max((1 - t) / 2, 0.0), 1.0)


def consistency(truth_acts, false_acts):
    """Cosine similarity of each pair's own difference to the mean direction."""
    diffs = truth_acts - false_acts
    mean_diff = diffs.mean(dim=0)
    mean_unit = mean_diff / torch.norm(mean_diff)
    sims = torch.nn.functional.cosine_similarity(diffs, mean_unit.unsqueeze(0), dim=1)
    return sims


def leave_one_out_accuracy(truth_acts, false_acts):
    """Rebuild the direction without pair i, then score pair i. Fraction correct."""
    n = truth_acts.shape[0]
    correct = 0
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        vec, _ = build_layer_vector(truth_acts[keep], false_acts[keep])
        risk_truth = risk_from(vec["unit"], vec["mid"], vec["half_span"], truth_acts[i])
        risk_false = risk_from(vec["unit"], vec["mid"], vec["half_span"], false_acts[i])
        if risk_truth < risk_false:
            correct += 1
    return correct / n


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, device=device, dtype="bfloat16", low_cpu_mem_usage=True
    )

    print(f"\nCaching activations for {len(PAIRS)} contrastive pairs...")
    truth_acts, false_acts, length_mismatches = collect_activations(model)

    if length_mismatches:
        print(
            f"\nNote: {len(length_mismatches)}/{len(PAIRS)} pairs tokenize to different "
            "lengths. Both are still read at their own final token, so the comparison\n"
            "holds, but the two final tokens sit at different sequence positions:"
        )
        for i, text, n_truth, n_false in length_mismatches:
            print(f"  pair {i + 1:2d}: {n_truth} vs {n_false} tokens | {text}")
    else:
        print("\nAll pairs tokenize to matching lengths (no positional confound).")

    print(f"\n{'Layer':>5} | {'||diff||':>9} | {'half_span':>9} | {'consistency':>11} | {'LOO acc':>7}")
    print("-" * 60)

    vectors = {}
    for layer in range(model.cfg.n_layers):
        vec, norm = build_layer_vector(truth_acts[layer], false_acts[layer])
        vectors[layer] = vec

        sims = consistency(truth_acts[layer], false_acts[layer])
        loo = leave_one_out_accuracy(truth_acts[layer], false_acts[layer])

        marker = "  <-- steering layer" if layer == REPORT_LAYER else ""
        print(
            f"{layer:5d} | {norm:9.3f} | {vec['half_span']:9.3f} | "
            f"{sims.mean():6.3f}+-{sims.std():.3f} | {loo:6.1%}{marker}"
        )

    print("\nConsistency = mean cosine similarity of each pair's own truth-minus-false")
    print("direction to the averaged direction (1.0 = all 50 pairs perfectly agree).")
    print("LOO acc = direction rebuilt from 49 pairs correctly ranks the held-out pair's")
    print("truthful sentence as less risky than its false one (50% = chance).")

    sims_10 = consistency(truth_acts[REPORT_LAYER], false_acts[REPORT_LAYER])
    print(
        f"\nLayer {REPORT_LAYER} detail: {(sims_10 > 0).sum().item()}/{len(PAIRS)} pairs "
        f"point the same way as the average (positive cosine); "
        f"weakest pair {sims_10.min():.3f}, strongest {sims_10.max():.3f}."
    )

    torch.save(vectors, OUT_PATH)
    print(f"\nSaved {len(vectors)} layer vectors to {OUT_PATH}")


if __name__ == "__main__":
    main()
