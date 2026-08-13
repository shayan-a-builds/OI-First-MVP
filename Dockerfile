FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights at build time so Cloud Run cold starts never
# hit the network -- only the local from_pretrained compute cost remains.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"

COPY Code/app.py Code/steering.py Code/steering_vectors.pt ./

ENV STREAMLIT_SERVER_HEADLESS=true \
    HF_HUB_OFFLINE=1

EXPOSE 8080
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
