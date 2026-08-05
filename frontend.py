import streamlit as st
import requests
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
# Replace this string with the active public URL printed by your Colab cell
NGROK_URL = "https://usage-ethically-divisible.ngrok-free.dev/" 

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Operational Interpretability Pipeline",
    page_icon="🧠",
    layout="wide"
)

# UI Header
st.title("🧠 Operational Interpretability Dashboard")
st.markdown("""
*Distributed System Pipeline*: **Dell Workstation (Streamlit Frontend)** ↔ **Google Colab (FastAPI Backend)**
""")

st.divider()

# Input Section
user_prompt = st.text_area(
    "Enter Prompt for Model Generation & Analysis:", 
    height=120, 
    value="Analyze feature activations in layer 3 during step-wise reasoning."
)

submit_btn = st.button("Generate & Analyze Risk", type="primary")

# Handling Execution
if submit_btn:
    if not user_prompt.strip():
        st.warning("Please enter a prompt before submitting.")
    else:
        # Format payload and clean endpoint URL
        target_url = f"{NGROK_URL.rstrip('/')}/generate"
        payload = {"text": user_prompt}

        with st.spinner("Communicating with Colab GPU backend..."):
            try:
                # Send Request
                response = requests.post(target_url, json=payload, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("Inference and interpretability telemetry retrieved!")
                    
                    # Section 1: Generated Text Output
                    st.subheader("1. Model Output")
                    st.info(data.get("generated_text", "No text returned."))
                    
                    st.divider()
                    
                    # Section 2: Metrics and Interpretability Visualization
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader("2. Hallucination Risk")
                        score = data.get("hallucination_risk_score", 0.0)
                        
                        # Display numeric metric and progress bar
                        st.metric(
                            label="Global Hallucination Risk Score", 
                            value=f"{score:.2f}",
                            delta=f"{'High Risk' if score > 0.5 else 'Low Risk'}",
                            delta_color="inverse"
                        )
                        st.progress(score)
                        
                    with col2:
                        st.subheader("3. Per-Layer Risk Profile")
                        layer_scores = data.get("layer_risk_array", [])
                        
                        if layer_scores:
                            # Format array into a DataFrame for visualization
                            df_layers = pd.DataFrame({
                                "Layer": [f"Layer {i+1}" for i in range(len(layer_scores))],
                                "Risk Score": layer_scores
                            }).set_index("Layer")
                            
                            st.line_chart(df_layers)
                        else:
                            st.write("No layer data available.")
                            
                else:
                    st.error(f"Backend Returned Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to reach backend at `{target_url}`. Check your `NGROK_URL` configuration.\n\nDetails: {e}")