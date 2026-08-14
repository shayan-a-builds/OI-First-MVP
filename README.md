# The Hallucination Risk Gauge

**This project exists to demonstrate my work in a research area called operational
interpretability.** The idea it works toward is that AI systems can be made deployable in
regulated, high stakes settings through a chain of three steps: understand what is
actually happening inside the model, translate that understanding into plain English
through an interface an ordinary person can use, then couple the result to the real laws
and compliance requirements that decide whether a system is allowed to run at all.

My hypothesis is that we already have the resources to do all three, and the work left is
joining the pieces into one puzzle rather than inventing new techniques. This repository
is the first MVP I built to test that in public. It is meant to show the approach is real
and tractable, not to be a finished product.

## Why any of this matters

A chatbot answers your question confidently and it happens to be wrong. You find out
later, if at all. And when you ask why it said that, the honest answer from everyone
involved, including the people who built it, is that nobody really knows.

That gap is why a hospital or a bank or a court cannot responsibly put one of these
systems into production, however good the demo looks.

This is a small working attempt at the first part of closing it. You type a prompt, and
a live gauge shows you a number pulled straight out of the model's internal state while
it writes: how true-looking or false-looking its activations are, on a scale built from
162 hand-checked sentence pairs. Then you can push that internal state along the same
axis the gauge measures and watch what changes.

**[Live demo](https://oi-first-mvp-111177658740.us-south1.run.app)** (scales to zero, so
give the first request a minute to wake up)

## Where this MVP sits on the three steps

Step one, looking inside the model, is done and validated. Step two, translating it into
something a person can read and act on, is what the app is. Step three, the compliance
and legal coupling, is the direction of travel and is deliberately not claimed as built.
Saying otherwise would defeat the point of the exercise.

## What holds up, and what does not

**The gauge works.** Take 162 pairs of sentences identical except for the answer, show
the model both, subtract the internal states, average across all of them. Tested by
rebuilding the ruler from 161 pairs and scoring the one it has never seen, 162 times
over, it ranks the held-out true sentence as less risky than its false twin 92% of the
time against a 50% baseline. That is real generalisation, not a tool agreeing with
itself.

**The steering mostly does not.** This project used to be named after a hallucination
kill switch. A sweep across 11 layers, 4 topics and 7 push strengths, 252 generations,
did not support that name, so the name changed rather than the claim quietly surviving.
Chemistry and biology facts never flipped anywhere at any strength. Capital city facts
did flip at a few layers, but often in the wrong direction. Expanding the ruler from 50
pairs to 162 made the measurement more honest and did not help the steering at all.

Negative results are still results, and that one is arguably the most useful thing here.
The measuring half of operational interpretability works. The control half is real but
narrow, and now characterised precisely enough to say how narrow and why.

## The full story

Everything above is the conclusion. The interesting part is how it was reached: the
hypothesis about fading corrections that turned out to be measurably wrong, the bug
where pushing toward truth produced more falsehood, the deployment that kept running out
of memory, the decision to hand-write 162 pairs instead of downloading a dataset, and
the sweep that took several hours and died twice before finishing.

That is all in **[Case Study.md](Case%20Study.md)**, written the same way as this, just
without leaving anything out.

## Limits worth knowing up front

It is a gauge, not a kill switch. Steering changes facts mainly on capital city style
prompts and elsewhere tends to shift tone while the fact stays put. Push strength has a
narrow usable band, roughly 5 to 10 either way, and the app ships with it set to 0 so
nothing is steered until you choose a strength yourself. That is deliberate: the sign is
unreliable at this layer, so pushing toward truth can produce a false answer and the
interface should not pretend otherwise. The steered answer is also a fresh generation
rather than an edit, so small wording differences are normal. And every number here comes
from one model, Qwen2.5-1.5B-Instruct, so none of it should be assumed to transfer.

## Running it locally

You need Python 3.12 or newer. The pinned dependencies only publish wheels for 3.12 and
up, and on anything older pip will quietly try to build ancient versions from source and
then fail in a confusing way.

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd Code
streamlit run app.py
```

On Windows, in PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd Code
streamlit run app.py
```

If PowerShell blocks the activate script, either use `venv\Scripts\activate.bat` from a
regular command prompt instead, or allow it for the current session with
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

Either way it opens in your browser. First launch downloads the model, which is about
3GB, so it takes a while. After that it is cached. It uses a GPU if it finds one and
falls back to CPU otherwise, which is slower but gives identical results.

```
Code/
  app.py                  the dashboard you interact with
  steering.py             loads the model, measures risk, applies steering
  contrastive_pairs.py    the 162 hand-checked true/false pairs
  build_vectors.py        builds the ruler and runs its validation
  alpha_sweep.py          the research script behind the 11-layer sweep
  steering_vectors.pt     the saved ruler
Dockerfile                packaging for Cloud Run
Case Study.md             the full build history
```

## Credits

Built with [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens),
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and
[Streamlit](https://streamlit.io/).
