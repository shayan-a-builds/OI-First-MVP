import pandas as pd
import streamlit as st

import steering

st.set_page_config(page_title="Hallucination Risk Gauge", page_icon="\U0001f9e0", layout="wide")

st.title("\U0001f9e0 Hallucination Risk Gauge")
st.markdown(
    "Operational interpretability demo: watch a live **hallucination risk** score read "
    f"straight out of the internal state of `{steering.MODEL_NAME}` as it writes, then "
    "push that internal state along the same truth/false axis and watch what changes."
)
st.info(
    "**What's proven and what isn't.** The gauge is the validated part: it reads a "
    "direction built from 162 hand-checked true/false sentence pairs across 9 topics, "
    "and it correctly ranks pairs it was never built from 92% of the time. The steering "
    "wheel is the **experimental** part. An 11-layer sweep found it does move the model, "
    "but not reliably in the direction you ask for: chemistry and biology facts never "
    "changed at all, and where capital-city facts did change it was often toward the "
    "false answer. Steering starts at 0 so you can try it yourself and see. "
    "This is a research instrument, not a hallucination fix.",
    icon="\U0001f9ea",
)
st.divider()

model = steering.load_model()
vectors = steering.load_vectors()

with st.expander("Advanced settings"):
    alpha = st.slider(
        "Steering strength (alpha)", -12.0, 12.0, steering.STEER_ALPHA, 0.25,
        help="Starts at 0, which applies no steering at all. Nothing visible happens until "
             "roughly +-5, the effect is clearest around +-8, and past roughly +-10 the "
             "output degrades into repetition. Positive aims at the truth anchor and "
             "negative at the false anchor, but the 11-layer sweep found that sign is "
             "unreliable here, so try both.",
    )
    max_new_tokens = st.slider("Max new tokens", 10, 80, 40, 5)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Prompt")
    prompt = st.text_area(
        "Enter a prompt where models tend to hallucinate or stall:",
        height=120,
        value="The capital of France is",
        help="The risk gauge works on any prompt. Steering visibly changes the *fact* "
             "mainly on capital-city style prompts -- on other topics you will usually "
             "see tone and framing shift while the fact itself stays put. That gap is a "
             "finding of this project, not a bug to work around.",
    )
    generate_clicked = st.button("Generate & Analyze Risk", type="primary")

if generate_clicked:
    if not prompt.strip():
        st.warning("Please enter a prompt before submitting.")
    else:
        with st.spinner("Generating and scoring..."):
            text, headline, layer_risks = steering.run(
                model, vectors, prompt, apply_steering=False, max_new_tokens=max_new_tokens
            )
        st.session_state.prompt = prompt
        st.session_state.output = text
        st.session_state.headline_risk = headline
        st.session_state.layer_risk_array = layer_risks
        st.session_state.pop("steered_output", None)

with col2:
    st.subheader("2. Hallucination Risk")
    if "headline_risk" in st.session_state:
        score = st.session_state.headline_risk
        st.metric(
            label=f"Layer {steering.STEER_LAYER} Risk Score",
            value=f"{score:.2f}",
            delta="High Risk" if score > 0.5 else "Low Risk",
            delta_color="inverse",
        )
        st.progress(score)

        st.caption(
            f"Per-layer risk profile. Steering is applied at layer {steering.STEER_LAYER}, "
            "the layer where the sweep found changes to the *fact itself* rather than just "
            "its wording. Layers 13-17 measure truth more accurately (layer 15 is the best "
            "at 92%) but steering them shifts tone and confidence while the fact stays put."
        )
        df = pd.DataFrame(
            {"Layer": list(range(len(st.session_state.layer_risk_array))),
             "Risk": st.session_state.layer_risk_array}
        ).set_index("Layer")
        st.line_chart(df)
    else:
        st.caption("Generate a response to see its risk profile.")

st.divider()

if "output" in st.session_state:
    st.subheader("3. Model Output")
    st.info(st.session_state.output)

    stop_clicked = st.button("\U0001f9ea Use AI Steering Wheel", type="primary")
    st.caption(
        "Set a steering strength in Advanced settings first (it starts at 0, which does "
        "nothing). Positive pushes toward the truth anchor, negative toward the false "
        "anchor, but the sweep found that sign is not reliable at this layer -- both "
        "directions can produce false claims. Watching that happen is the experiment."
    )

    if stop_clicked and alpha == 0:
        st.warning(
            "Steering strength is 0, so this will reproduce the unsteered answer. "
            "Open Advanced settings and move the slider past +-5 to see an effect."
        )

    if stop_clicked:
        with st.spinner("Injecting steering vector and regenerating..."):
            steered_text, steered_headline, steered_layer_risks = steering.run(
                model, vectors, st.session_state.prompt,
                apply_steering=True, alpha=alpha, max_new_tokens=max_new_tokens,
            )
        st.session_state.steered_output = steered_text
        st.session_state.steered_headline_risk = steered_headline
        st.session_state.steered_layer_risk_array = steered_layer_risks

    if "steered_output" in st.session_state:
        before_col, after_col = st.columns(2)
        with before_col:
            st.markdown("**Before (unsteered)**")
            st.write(st.session_state.output)
            st.metric("Risk", f"{st.session_state.headline_risk:.2f}")
        with after_col:
            st.markdown("**After (steered)**")
            st.write(st.session_state.steered_output)
            delta = st.session_state.steered_headline_risk - st.session_state.headline_risk
            st.metric("Risk", f"{st.session_state.steered_headline_risk:.2f}", delta=f"{delta:+.2f}", delta_color="inverse")
