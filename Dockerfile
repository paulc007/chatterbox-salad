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

# Copy voice reference files
COPY voices/ ./voices/

# Copy app
COPY app.py .

# SaladCloud requires IPv6, so bind to all interfaces
CMD ["uvicorn", "app:app", "--host", "::", "--port", "8000"]
