FROM python:3.11-slim

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (explicit --break-system-packages for slim image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Chatterbox-Turbo model from Hugging Face during build
# Saves 2-10 min of GPU time on every cold start
ARG HF_TOKEN
ENV HF_HOME=/app/hf_cache
RUN python -c "\
import os;\
from huggingface_hub import snapshot_download;\
print('Downloading ResembleAI/chatterbox-turbo...');\
snapshot_download('ResembleAI/chatterbox-turbo', token=os.environ.get('HF_TOKEN'));\
print('Model cached at', os.environ['HF_HOME'])"

# Copy voice reference files
COPY voices/ ./voices/

# Copy app
COPY app.py .

# SaladCloud requires IPv6, so bind to all interfaces
CMD ["uvicorn", "app:app", "--host", "::", "--port", "8000"]
