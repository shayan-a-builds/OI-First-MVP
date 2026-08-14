"""Sweeps steering strength (alpha) across layers for the controller in steering.py.

The controller sets each newly-generated token's projection onto the layer's
truth<->false axis to exactly `alpha * half_span` (see steering.py's module
docstring). This script is what produced the 11-layer causal sweep reported
in Case Study.md: 4 topics x 7 strengths x layers 10-20, 252 generations.

Layers 0-9 and 21-27 are deliberately excluded. build_vectors.py's own
validation shows layers 0-9 carry no usable direction (leave-one-out at or
below chance, consistency near zero) and layers 21-27 degrade steadily as
the residual stream specialises for next-token prediction. Layers 10-20 are
where a strong, consistent truth direction coincides with an intervention
point early enough to shape how the answer forms rather than just what gets
printed. See Case Study.md, "Layer selection", for the full reasoning.

Prints every generation with its risk score so the working range can be
picked by inspection. This is a one-off research script like
build_vectors.py, not something meant to run unattended in CI.

`steering.STEER_LAYER` is temporarily overridden per layer for the duration
of the sweep and the module's real default is left untouched.
"""

import sys

# Generations can contain characters the default Windows console encoding
# cannot represent, which otherwise kills a multi-hour run mid-sweep.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import steering

LAYERS = list(range(10, 21))

ALPHAS = [-10, -6, -3, 0, 3, 6, 10]

PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "The chemical symbol for gold is",
    "The organ that pumps blood is the",
]

MAX_NEW_TOKENS = 20


def main():
    model = steering.load_model()
    vectors = steering.load_vectors()

    for layer in LAYERS:
        steering.STEER_LAYER = layer
        print(f"\n{'#' * 70}\nLAYER {layer}\n{'#' * 70}")
        for prompt in PROMPTS:
            print(f"\n{'=' * 70}\nPROMPT: {prompt!r}\n{'=' * 70}")
            for alpha in ALPHAS:
                text, headline, _ = steering.run(
                    model, vectors, prompt, apply_steering=True, alpha=alpha,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                print(f"alpha={alpha:+5.1f}  risk={headline:.3f}  text={text!r}")

    print("\nDone.")


if __name__ == "__main__":
    main()
