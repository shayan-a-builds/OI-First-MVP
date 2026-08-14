# Case Study: The Hallucination Risk Gauge

## Purpose

This system was built to demonstrate my work in a research area called operational
interpretability, rather than to ship a product. The idea it works toward is that an AI
system becomes deployable in regulated, high stakes environments through three connected
steps: establishing what is happening inside the model, translating that into plain
language through an interface a non-specialist can operate, and coupling the result to
the legal and compliance requirements that govern whether such a system may be used at
all.

My working hypothesis is that the components required for all three already exist, and
that the outstanding work is integration rather than invention. This MVP is my first
public test of that. It implements the first two steps and treats the third as stated
direction rather than delivered capability.

Every scoping decision recorded below follows from that purpose. Where the evidence did
not support a claim, the claim was withdrawn rather than the evidence being presented
selectively, since an overstated result would undermine the argument the project exists
to make.

## Scope of this document

A technical record of how the system was built, what was measured, and what the evidence
supports. The README covers the summary. This document covers the method, the results in
full, and the defects found along the way.

Model throughout is Qwen2.5-1.5B-Instruct, accessed via TransformerLens. All numbers
below come from that model and should not be assumed to transfer to others.

## Method

The system rests on a single idea. Two sentences that are identical except for their
final answer, one true and one false, produce two different internal states in the
model. Subtracting one from the other gives a direction in activation space pointing
from false-looking toward true-looking. Averaged over many pairs, the components
specific to any individual fact cancel and the shared component survives.

That direction serves two purposes. It measures, by projecting a new activation onto
it, and it steers, by modifying activations during generation.

Scoring stores two reference points alongside the direction: the midpoint between the
true and false anchors, and the distance from that midpoint to the true anchor. A live
activation is projected onto the direction, positioned relative to those two points,
and mapped into a 0 to 1 range where 0 matches the true anchor and 1 matches the false
anchor.

No gradient descent is involved at any stage. There is no trained classifier. The
distinction matters because it determines what fixes are available when results
disappoint. Poor performance here is addressed by improving the variety of the
contrastive data, not by training longer on more of it.

## Building the ruler

The first version used one contrastive pair. Sufficient to demonstrate a mechanism,
insufficient to claim generality. It was rebuilt twice.

**Design rules.** Each pair is minimal, differing only in the answer, so that sentence
length and topic do not contaminate the difference vector. Wrong answers are
type-matched, a city substituted for a city and a year for a year. An early version
compared Paris against Banana, which conflates being false with being a category error.
Only uncontested facts are included, excluding items such as the longest river in the
world where the answer is genuinely disputed. Topics are spread across geography,
physics and chemistry, biology, astronomy, mathematics, history, language and
technology.

**Validation.** Two checks are computed at build time. Consistency measures how closely
each individual pair's own difference vector aligns with the average, indicating whether
the pairs share one axis or many unrelated ones. Leave-one-out accuracy rebuilds the
direction from all pairs but one and tests whether it correctly ranks the held-out
pair's true sentence as less risky than its false counterpart. The second is the
substantive test, being the only one that evaluates performance on data the ruler was
not built from.

**Fifty pair results.**

| Layer | Direction length | Consistency | Leave-one-out |
|---|---|---|---|
| 10 | 3.364 | 0.336 | 84.0% |
| 13 | 5.803 | 0.401 | 98.0% |
| 15 | 9.286 | 0.505 | 100.0% |
| 17 | 10.080 | 0.469 | 96.0% |
| 27 | 31.765 | 0.353 | 64.0% |

Truth separates most cleanly in the middle of the network, peaking at layer 15, and
degrades toward the output layers.

One immediate consequence was that the averaged layer 10 direction is roughly 4.5 times
shorter than the original single-pair vector, 3.364 against 15.099, invalidating all
existing strength calibration.

**Expansion to 162 pairs.** Ten of the fifty pairs, a full fifth, used the same capital
of X is Y template, and that was also the only prompt shape where steering reliably
worked. To test whether narrow phrasing was the limiting factor, the set was expanded to
162 pairs with roughly equal coverage per topic, 18 to 22 each instead of a range from 3
to 10, and four to six distinct sentence structures per topic rather than one repeated
template.

An existing public dataset was evaluated first. The Geometry of Truth corpus was
rejected on two grounds: its cities file repeats a single rigid template, reproducing
the exact problem under investigation, and its more varied files are not organised as
matched true and false versions of the same statement. That pairing is the mechanism by
which topic and length cancel before averaging, and without it the confounds the design
rules exist to remove are reintroduced.

Rebuilding from 162 pairs reduced leave-one-out accuracy at every layer:

| Layer | 50 pairs | 162 pairs |
|---|---|---|
| 10 | 84.0% | 75.9% |
| 13 | 98.0% | 90.7% |
| 15 | 100.0% | 92.0% |
| 17 | 96.0% | 87.0% |

This reflects a harder test rather than a worse direction. When a fifth of the pairs
share one template, generalising to a held-out pair of that same shape is close to
trivial. Across 162 pairs spanning many structures, the direction must generalise across
genuine phrasing variety. Layer 15 remained the strongest measuring layer under the
stricter test, indicating the earlier result was not an artifact of narrow data.

**Known confound.** 52 of the 162 pairs tokenise to different lengths, placing their
final tokens at different sequence positions. Each is still read at its own final token
so the comparison holds, but the positional difference is a residual confound, resolvable
with length-matched distractors.

## Steering mechanism

**Initial implementation.** A fixed vector added to activations at every token. This
produced a correction that visibly weakened over longer generations.

**Hypothesis tested and rejected.** The standard explanation is that residual stream
magnitude grows with sequence position, making a constant addition proportionally
smaller over time. This was measured directly across 60 token generations on three
prompts. Correlation between position and magnitude was inconsistent at 0.503, 0.320 and
-0.104. The ratio between final and first generated position remained within 4% of 1.0
in all cases. The correction held steady at 6 to 7% of the signal throughout. Dilution is
therefore not the mechanism. The more likely explanation is accumulating commitment to
previously generated tokens, against which a constant push loses ground regardless of
its relative magnitude.

**Closed-loop replacement.** The correction now reads each token's current position on
the truth axis and corrects it to a target position, recalculated at every token from
the actual current state. This cannot fade in the way an accumulated blind offset can.

**Defect: correction applied to prompt encoding.** The first closed-loop version pushed
toward truth and produced more false output than pushing toward falsehood. The cause was
that the correction was applied to all sequence positions including the prompt, altering
the model's representation of the question rather than influencing the answer.
Restricting it to newly generated positions corrected the direction.

**Defect: scoring window.** Risk was originally read from the single final token of the
generated text. Where generation stops at a token limit, that token is frequently a
mid-word fragment carrying little information about the truth of the answer. Scoring now
averages per-token risk across the last several generated tokens.

**Calibration.** With the closed-loop mechanism and the new ruler, the effect becomes
visible only past roughly 5 in either direction, peaks near 8, and degrades into
repetition past 10. The previous default of 1.0 produced no visible effect, and the UI
slider maximum of 5 fell entirely within the dead zone. The slider range was widened to
plus or minus 12.

The shipped default is nevertheless 0, meaning no steering is applied until the user
chooses a strength. End to end verification at a default of 8.0 produced three
consecutive false claims on the application's own example prompt, with the risk gauge
correctly rising from 0.920 to 1.000. Given that the sweep found the sign of the effect
to be unreliable at this layer, any non-zero default would ship an interaction whose
label promises a direction the evidence does not support. Starting neutral makes the
uncertainty the user's to explore rather than something the interface asserts.

**Defect: zero strength was not a no-op.** Verifying that the new default of 0 left
output untouched showed that it did not. The controller targets a projection of alpha
multiplied by the half span, so a strength of 0 pins every generated token to the exact
midpoint between the true and false anchors, which is an active intervention rather than
an absent one. Generation diverged from the unsteered baseline within fifteen tokens.
The controller now skips hook registration entirely when strength is 0, giving the
slider a genuine off position. The discontinuity this introduces at 0 has no practical
consequence, since the dead zone below plus or minus 5 means nearby values produce no
visible change either.

## Layer selection

The model has 28 layers. Building the ruler produces validation scores for all of them,
which constrains where steering is worth testing before any generation is run.

| Layers | Direction length | Consistency | Leave-one-out |
|---|---|---|---|
| 0 to 6 | 0.22 to 0.65 | 0.07 to 0.11 | 17% to 27% |
| 7 to 9 | 1.31 to 1.52 | 0.18 to 0.19 | 41% to 58% |
| 10 to 20 | 2.73 to 13.50 | 0.27 to 0.47 | 75% to 92% |
| 21 to 27 | 15.59 to 34.16 | 0.39 to 0.44 | 65% to 73% |

**Early layers contain no usable direction.** Consistency near zero indicates the pairs
share no common axis. Leave-one-out falls between 17% and 27%, substantially below the
50% chance baseline, which typically indicates a measurement tracking something real but
not the labelled property. The probable cause is that these layers have not yet
assembled a proposition. At the final token of the capital of France is Paris, an early
layer largely represents the token Paris rather than the claim about Paris. Averaging
Paris minus Madrid against Au minus Ag and 160 other token-identity differences yields
noise, since the only shared property is that the words differ.

**Late layers represent output rather than meaning.** Accuracy declines steadily from
76.5% at layer 20 to 65.4% at layer 27, with widening spread across pairs. Direction
length grows from 2.73 at layer 10 to 34.16 at layer 27, which reflects normal residual
stream growth rather than a stronger signal. The interpretation is that the final layers
convert conclusions into a distribution over vocabulary tokens, shifting the represented
quantity from whether a claim is true toward which token comes next.

A second consideration applies at the deep end. Intervening immediately before the output
is functionally close to editing the output distribution, which overrides the answer
without altering the reasoning that produced it and yields no mechanistic insight.

Layers 10 to 20 are where a strong, consistent truth direction coincides with an
intervention point early enough to influence how the answer forms. Testing all 28 layers
would have roughly tripled a multi-hour run to confirm what the validation scores already
establish at both ends.

## The eleven layer sweep

Layers 10 and 15 had been tested causally and behaved differently from each other, which
is insufficient evidence for a general claim. Every layer from 10 to 20 was therefore
tested identically: four topics, seven strengths from -10 to +10, twenty token
generations. 252 generations in total.

**Results.**

Chemistry and biology facts did not change at any layer, at any strength, in either
direction, across all 126 generations covering those topics. Gold remained Au and the
heart remained the heart.

Arithmetic did not reliably change with the expanded ruler.

Capital city facts changed at layers 10, 13, 16 and 19, approximately every third layer.
In nearly all cases the change occurred at positive strength, the direction intended to
push toward truth. Layer 13 at +10, layer 16 at +6, and layer 19 at +3 and +6 all
produced the false claim that Paris is the capital of the United States. The sign is
therefore unreliable, consistently, across multiple layers.

From approximately layer 14 onward the dominant effect is neither fact substitution nor
confident elaboration but self-negation without substitution, producing output of the
form no, this is false, frequently followed by a restatement of the correct fact. The
tone shift observed earlier at layer 15 is representative of the deeper layers rather
than exceptional.

**Interpretation.** Single-direction activation steering, as implemented here, does not
provide reliable, correctly-signed, cross-topic factual correction at any layer in the
usable range of this model. The conclusion rests on eleven layers and two independently
constructed rulers.

## Infrastructure

**Hosting.** Streamlit Community Cloud was unusable because TransformerLens peaks near
4GB during model loading against a container limit of approximately 1GB. Reducing model
precision or size was rejected, since that degrades the quality the measurements depend
on. Hugging Face Spaces with a Docker SDK is a paid tier. Google Cloud Run was selected
for its permanent free tier, scale to zero behaviour, and container-based deployment. The
service runs with 16GB.

**Memory measurement.** Peak resident set size was sampled on a background thread every
50ms during load and generation, recording 9.69GB peak against approximately 7.1GB steady
state. The transient peak is invisible to measurements taken after loading completes,
which is what made the original hosting failure difficult to diagnose.

**Defect: Python version mismatch.** The initial Cloud Build failed reporting no matching
distribution for numpy 2.5.1. The base image was python:3.11-slim while the pinned
dependencies publish wheels only for 3.12 and above. Absent a compatible wheel, pip
searched backwards through source releases before failing, producing an error message
that identifies the wrong cause. Resolved by moving the base image to python:3.14-slim.

**Defect: concurrent hook state.** Production raised a KeyError on a layer name during
steering that did not reproduce locally. The model is cached as a single object shared
across requests, and TransformerLens hooks mutate that shared state, so overlapping
requests can interleave hook registration and corrupt each other's activation cache. A
process-wide lock now serialises generation and scoring. This remains a hypothesis-driven
fix rather than a confirmed root cause, though sharing mutable hook state across
concurrent requests is unsafe independently of this specific failure.

**Sweep reliability.** The eleven layer run failed twice before completing. The first
attempt was terminated by an environment restart. The second crashed at layer 17 when
generated output contained characters the default Windows console encoding could not
write to the log. Output encoding was forced to UTF-8 and completed layers were preserved
so only the remainder required rerunning.

## Conclusions

The measurement component is validated. A truth-correlated direction can be extracted
from a running model, shown to generalise to facts it was not built from at 92% against a
50% baseline, and presented to a non-specialist as an actionable number.

The control component is real but narrow. Steering demonstrably moves the model and does
so in structured, explicable ways, but does not deliver reliable factual correction
across topics at any tested layer. This is characterised precisely enough to state the
boundary and the probable reasons for it.

The project is accordingly positioned as a risk gauge with experimental steering rather
than a hallucination kill switch, which was its original name and is not supported by the
evidence.

## Future work

**Per-topic steering vectors.** A direction with causal power over capital city facts and
none over chemistry facts is consistent with different fact categories being retrieved
through different internal circuitry, which a single averaged direction cannot represent.
Constructing one vector per topic would test this directly.

**Multi-layer intervention.** If fact retrieval is distributed across a span of layers
rather than concentrated at one, single-point intervention may provide insufficient
leverage regardless of the point chosen. Simultaneous intervention across several layers
is the natural next experiment.

Neither is attempted in this version. Both are scoped follow-up work.
