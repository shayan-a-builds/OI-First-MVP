import torch
from transformer_lens import HookedTransformer

# 1. Load your local model
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    device="cuda",
    torch_dtype=torch.bfloat16
)

# 2. Run factual vs non-factual prompts
_, cache_true = model.run_with_cache("The capital of France is Paris.")
_, cache_false = model.run_with_cache("The capital of France is Banana.")

# 3. Extract internal activations at Layer 10 (Residual Stream)
act_true = cache_true["blocks.10.hook_resid_post"][0, -1, :]
act_false = cache_false["blocks.10.hook_resid_post"][0, -1, :]

# 4. Measure the difference between the two states
difference_vector = act_false - act_true
difference_magnitude = torch.norm(difference_vector).item()

print(f"\n[RESEARCH RESULT]")
print(f"Layer 10 Activation Difference Magnitude: {difference_magnitude:.4f}")
print("You just mathematically measured internal state divergence!")

