FROM python:3.11-slim

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Chatterbox-Turbo model from Hugging Face during build
ARG HF_TOKEN
ENV HF_HOME=/app/hf_cache
RUN python -c "\
import os;\
from huggingface_hub import snapshot_download;\
print('Downloading ResembleAI/chatterbox (full 0.5B)...');\
snapshot_download('ResembleAI/chatterbox', token=os.environ.get('HF_TOKEN'));\
print('Model cached at', os.environ['HF_HOME'])"

# Empty voices dir — populated at runtime via POST /upload-voice
RUN mkdir -p voices

# Copy app
COPY app.py .

# SaladCloud requires IPv6, so bind to all interfaces
CMD ["uvicorn", "app:app", "--host", "::", "--port", "8000"]
