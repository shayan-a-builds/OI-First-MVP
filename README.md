---
title: Hallucination Kill-Switch
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.61.0"
app_file: Code/app.py
pinned: false
---

# 🧠 The Hallucination Kill-Switch

You type a prompt into a chatbot. It answers confidently. It's wrong. You have no way
of knowing that until after the fact, and even then, nobody — not you, not the company
that built it — can point to *why* it said what it said, in a way that would hold up
to a customer, a regulator, or a judge.

This project is a small, working proof that it doesn't have to be that way. You type a
prompt, watch a live gauge that shows how "confident vs. confused" the AI's internal
state actually is, and then — if it's about to say something wrong — press a button
that reaches into the model mid-thought, physically nudges it back toward the truth,
and lets you watch the answer change in real time.

It's a tiny demo, built around one fact (a French capital city), on a small
open-source model. But the mechanism behind it is real, and it's the seed of a much
bigger research question, explained below.

## The research question this is really about

The field this belongs to is called **operational interpretability**, and the goal is
this: build AI systems that can explain themselves in plain English, and hold up that
explanation in a courtroom or a compliance audit if they ever need to.

Right now, that's not how AI works. A model is a black box — you put text in, text
comes out, and if it says something dangerous, biased, or made-up, the honest answer
to "why did it say that?" is "nobody really knows." That's a legal and safety problem,
not just an engineering one. A hospital, a bank, or a court can't responsibly deploy a
system it cannot explain or control.

Operational interpretability is about closing that gap in three steps, and this
project is a working (if small) example of all three:

1. **Look inside the AI's brain.** Actually go into the model's internal math while
   it's "thinking" and find the specific place where a concept like "this is true"
   vs. "this is false" lives.
2. **Translate that for humans.** Turn that raw internal math into something a normal
   person can look at and understand — a gauge, a number, a button — not a wall of
   numbers only an engineer can read.
3. **Wire it into something usable in the real world.** Give a human a real, visible
   control over the system, so that if something goes wrong, there's an actual record
   of "the system detected this, and here's what it did about it" — the beginning of
   the kind of audit trail that real-world compliance and legal accountability
   require.

The button in this app is literally labeled **"Stop Hallucinating!"** instead of
something like "apply directional ablation coefficient" — that's not a cute UI choice,
it's the second step of the research (translating internal AI math into something a
regular person can act on) actually being demonstrated, not just described.

## How an AI actually "decides" something (the part most people get wrong)

Before any of the code made sense, the first thing to unlearn was thinking of an AI's
"thought process" like a race — as if two ideas (say, "Paris" and "Banana") are cars
speeding toward the finish line, and whichever one arrives first is what the model
says.

That's not what happens. Every part of the model processes everything at the exact
same time, in lockstep — there's no "faster" or "slower" idea in there. A much better
mental picture is a map. Every "thought" the model has is really just a point (a set
of coordinates) somewhere on a giant, high-dimensional map. Ideas that mean similar
things sit near each other on that map, the way "Paris" and "France" and "Eiffel
Tower" would cluster together, far away from "Banana" or "Ecuador."

"Steering" a model, then, isn't about changing how fast an idea wins a race — it's
about physically shoving the model's current coordinates out of one neighborhood on
that map and into another. Push its internal position out of the "wrong answer"
neighborhood and into the "right answer" neighborhood, and the words it produces next
change accordingly.

One more important nuance: this isn't a light switch, it's a dial. Push too gently and
nothing changes. Push exactly right and you get a clean correction. Push too hard and
you don't get "extra truthful" — you get broken, garbled nonsense, because you've
shoved the model's coordinates completely off the map. Finding that reliable "sweet
spot" is most of what real interpretability research actually is — not writing an
if/then rule, but finding a range of settings that works reliably, the way a doctor
finds a safe and effective drug dosage rather than an on/off switch.

## What the app actually does, step by step

1. **You type a prompt** — ideally something a small AI model tends to get wrong or
   get weirdly evasive about.
2. **It answers, and we measure its internal state.** Behind the scenes, we check
   where the model's "thoughts" landed on the true/false map described above, and
   turn that into a single number between 0 and 1: 0 means its internal state looked
   exactly like it does when it's about to say something true, 1 means it looked
   exactly like it does when it's about to say something false. That number is the
   gauge you see on screen.
3. **You press "Stop Hallucinating!"** This regenerates the answer, but this time we
   reach into the model's internal map, at the exact spot we identified, and add a
   nudge in the "truthful" direction before the model finishes forming its answer.
   Nothing about the model itself is changed or damaged — we're not editing its
   memory or retraining it, we're just leaning on its train of thought for one
   answer, one time. The next question starts completely fresh.
4. **You compare before and after** — the original answer and risk score side by
   side with the steered answer and its new, hopefully lower, risk score.

## The actual story of how this got built (mistakes included)

This didn't come together cleanly on the first try, and the mistakes are as much a
part of the research record as the parts that worked.

**Step one was just making sure the basic wiring worked at all** — could a script on
this laptop successfully load an open-source AI model, run it on the GPU, and read
out its internal numbers without crashing. Boring, but necessary — you can't debug
subtle AI behavior if you're not even sure the plumbing works.

**Then came the actual science: measuring the difference between "true" and
"false" inside the model.** The approach was to show the model two nearly identical
sentences — one true ("The capital of France is Paris"), one false ("The capital of
France is Banana") — and take a snapshot of its internal state for each. Subtracting
one snapshot from the other gives you a direction: a mathematical arrow pointing from
"the model believes something false" toward "the model believes something true."
That arrow is the whole engine behind this project.

**Mistake #1: almost overwriting the code with itself.** Early on, a script got saved
with the wrong file extension — the format PyTorch uses to save raw number data
(`.pt`) instead of the format for actual code (`.py`). Running it as written
would have quietly overwritten real code with binary number soup. Caught before
running it, but a good reminder that small, boring mistakes like a wrong file
extension can silently destroy real work.

**Mistake #2: the nudge that did absolutely nothing.** The first real attempt at
injecting that "truthful direction" arrow back into the model produced identical
output no matter how hard the nudge was pushed — same answer at every strength
tested. It turned out the way the nudge was being attached to the model was being
silently ignored by the function that generates text; the model was never actually
receiving it. The fix was attaching the nudge differently, so it stayed active for
the entire duration of the model "thinking through" its answer, not just for a single
instant that got skipped over. This is the kind of bug that's dangerous precisely
because it fails silently — everything *looks* like it's working, and you only
notice something's wrong because the results feel suspiciously identical.

**Once the nudge actually worked, the next step was finding that "sweet spot"
dial setting.** Testing a range of nudge strengths on the same prompt showed a very
clear pattern: too weak, and nothing happens. A moderate, correct amount, and the
model's wrong answer flips cleanly to the right one, with the rest of the sentence
still reading naturally. Push further, and the sentence starts to wander
semantically. Push further still, and the model starts leaking totally unrelated
content — bits of math notation, characters from other languages, things that had
nothing to do with the prompt. Push it to the extreme, and the output degrades into
complete gibberish. That middle "clean correction" zone is what the app defaults to
today.

**Then, to prove this wasn't a fluke, the nudge was tested in reverse.** If pushing
the model's internal state toward "truthful" makes it say more true things, then
pushing the exact same lever the opposite direction, on purpose, should make it say
*false* things — deliberately induced hallucination. It did, reliably. That's
actually a really important piece of evidence: if a single lever can reliably push a
model's behavior in either direction along an axis, that's strong proof you've found
a real mechanism inside the model, not a coincidence or a side effect of something
else.

**Mistake #3: a wrong mental model of how the nudges "add up."** After testing small
positive and small negative nudges back to back and seeing similar-looking results,
it was tempting to think of it like footsteps — push forward a little, push back a
little, and you've basically cancelled yourself out and ended up back where you
started. That's not actually what's happening. Every single answer the model gives
starts completely fresh — it has no memory of the previous question or the previous
nudge. The reason small nudges in either direction can look similar to the unsteered
answer isn't cancellation, it's that the model is choosing its single most-confident
next word every time, like a coin balanced on its edge — a small nudge can tilt the
underlying odds without necessarily being enough to actually tip the coin over to a
different word. A bigger nudge is needed to reliably flip which word wins.

**Finally, this got turned from a terminal window full of numbers into something a
normal person could actually use** — a real on-screen gauge, a plain-language button,
and a side-by-side before/after comparison, instead of reading raw print statements
off a script. That's the "translate it for humans" step of operational
interpretability, made literal.

**One more late decision worth mentioning:** the original plan was to run the AI
model on a free Google Colab GPU in the cloud, with this local laptop talking to it
over a temporary tunnel service. That's a reasonable way to get free GPU power, but
it adds a lot of extra failure points for something meant to be shown live or shared
publicly — the free cloud GPU can disconnect, the tunnel's web address changes every
time it restarts. Since this project is meant to be open-source and something anyone
can actually click on and try, it was simplified into one single, self-contained app
that can run entirely on its own, and be hosted permanently for free on a platform
built specifically for sharing exactly this kind of AI demo — no separate server,
no tunnel, no cloud GPU dependency required.

## How the "risk gauge" number is actually calculated

No magic here — it's a ruler. We already have two known reference points: what the
model's internal state looks like when it's about to say something true, and what it
looks like when it's about to say something false. Those two points define a line.
For any new prompt, we check where the model's current internal state lands on that
same line: right on top of the "true" reference point scores 0, right on top of the
"false" reference point scores 1, and anywhere in between is scored proportionally.

Before trusting this on real prompts, the ruler was checked against its own two
reference points — and it correctly scored the "true" sentence a 0.0 and the "false"
sentence a 1.0, exactly as it should. That's not a hugely surprising result (the ruler
was built from those two points, so of course it agrees with itself) but it's the
basic sanity check every measurement tool needs before you trust it on anything new.

## Where this project is honest about its limits

Real research credibility means being upfront about what this does *not* prove yet:

- **The ruler is built from a single example** — one true fact and one false version
  of it. That's enough to prove the mechanism works, but it is not yet a
  general-purpose lie detector that would reliably work across arbitrary topics. A
  more rigorous version of this would build that same ruler from hundreds of
  true/false pairs across many subjects, not one.
- **Only one specific spot inside the model has actually been proven to work as a
  control lever** — the "sweet spot" dial testing was only done at that one location.
  The app can *measure* the same kind of signal at other locations inside the model
  for comparison, but hasn't proven that nudging those other locations would work the
  same way.
- **The "after" answer is a brand new answer, not an edit.** Pressing the button
  regenerates the response from scratch under the nudge, rather than correcting the
  specific wrong word in the original sentence — so besides fixing the fact in
  question, small other wording differences between the before and after answer are
  normal and expected.

Those limitations are exactly the next research steps: broadening the ruler beyond
one example, testing whether the same lever works at other locations inside the
model, and building the kind of rigorous, reproducible audit trail that could
eventually stand up to real legal or regulatory scrutiny — not just a laptop demo.

## Try it yourself

A live version of this is hosted for free, permanently, so anyone can try it without
installing anything — link goes here once the Hugging Face Space is live.

### Running it on your own machine

If you'd rather run it locally:

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cd Code
streamlit run app.py
```

It works with or without a GPU — it automatically uses one if it finds it, and falls
back to your regular processor otherwise (slower, but the results are identical).

### Repo layout

```
Code/
  app.py                 the on-screen dashboard (what you actually interact with)
  steering.py             the engine: loads the model, measures risk, applies the nudge
  build_vectors.py        the one-time script that built the "ruler" described above
  steering_vectors.pt     the saved ruler itself, so the app doesn't rebuild it every time
requirements.txt          the list of software this project depends on
```

## Credits

Built with [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens),
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), and
[Streamlit](https://streamlit.io/).
