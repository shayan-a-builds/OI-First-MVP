import pandas as pd
import streamlit as st

import steering

st.set_page_config(page_title="Hallucination Kill-Switch", page_icon="\U0001f9e0", layout="wide")

st.title("\U0001f9e0 Hallucination Kill-Switch")
st.markdown(
    "Operational interpretability demo: watch a live **hallucination risk** score for "
    f"`{steering.MODEL_NAME}`, then hit the kill-switch to steer the model's internal "
    "state back toward truthfulness in real time."
)
st.divider()

model = steering.load_model()
vectors = steering.load_vectors()

with st.expander("Advanced settings"):
    alpha = st.slider(
        "Steering strength (alpha)", -2.0, 5.0, steering.STEER_ALPHA, 0.25,
        help="1.0 is the validated 'Goldilocks zone' from the alpha sweep. "
             "Negative values intentionally push the model toward hallucination.",
    )
    max_new_tokens = st.slider("Max new tokens", 10, 80, 40, 5)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Prompt")
    prompt = st.text_area(
        "Enter a prompt where models tend to hallucinate or stall:",
        height=120,
        value="The capital of France is",
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
            f"Per-layer risk profile (layer {steering.STEER_LAYER} is the only "
            "causally-validated steering point; other layers are shown for comparison)."
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

    stop_clicked = st.button("\U0001f6d1 Stop Hallucinating!", type="primary")

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
