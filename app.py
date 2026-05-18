"""
Chatterbox-TTS API for SaladCloud GPU deployment.
Exposes /generate endpoint with multi-voice support.
"""

import io
import os
import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ---------- globals ----------
MODEL = None
VOICES_DIR = os.path.join(os.path.dirname(__file__), "voices")
AVAILABLE_VOICES: list[str] = []


def find_voices() -> list[str]:
    """Scan voices dir for .wav files, return list of voice names."""
    if not os.path.isdir(VOICES_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(VOICES_DIR)
        if f.lower().endswith(".wav")
    )


def get_ref_path(voice: str) -> str:
    return os.path.join(VOICES_DIR, f"{voice}.wav")


# ---------- lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, AVAILABLE_VOICES
    AVAILABLE_VOICES = find_voices()
    print(f"Voices available: {AVAILABLE_VOICES}")
    print("Loading Chatterbox-Turbo model...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = ChatterboxTurboTTS.from_pretrained(device=device)
    print(f"Model loaded on {device}")
    yield
    # cleanup
    del MODEL


app = FastAPI(title="Chatterbox-TTS", lifespan=lifespan)


# ---------- request model ----------
class GenerateRequest(BaseModel):
    text: str
    voice: str = "FR"  # default voice


# ---------- health probes (required by SaladCloud) ----------
@app.get("/started")
async def startup_probe():
    return {"status": "started"}


@app.get("/live")
async def liveness_probe():
    if MODEL is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "live"}


@app.get("/ready")
async def readiness_probe():
    if MODEL is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready"}


# ---------- list voices ----------
@app.get("/voices")
async def list_voices():
    return {"voices": AVAILABLE_VOICES}


# ---------- generate ----------
@app.post("/generate")
async def generate(req: GenerateRequest):
    if MODEL is None:
        raise HTTPException(503, "model not loaded")

    if req.voice not in AVAILABLE_VOICES:
        raise HTTPException(
            400,
            f"Unknown voice '{req.voice}'. Available: {', '.join(AVAILABLE_VOICES)}"
        )

    ref_path = get_ref_path(req.voice)
    if not os.path.isfile(ref_path):
        raise HTTPException(404, f"Reference file missing: {ref_path}")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is required")

    # Generate
    wav = MODEL.generate(text, audio_prompt_path=ref_path)

    # Encode to WAV bytes
    buf = io.BytesIO()
    ta.save(buf, wav, MODEL.sr, format="wav")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")
