"""
Chatterbox-TTS API for SaladCloud GPU deployment.
Exposes /generate endpoint with multi-voice support.
Voices are uploaded at runtime via /upload-voice — no baked-in refs.
"""

import io
import os
import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
    os.makedirs(VOICES_DIR, exist_ok=True)
    AVAILABLE_VOICES = find_voices()
    print(f"Voices available at startup: {AVAILABLE_VOICES or '(none — upload via POST /upload-voice)'}")
    print("Loading Chatterbox (full 0.5B) model...")
    from chatterbox.tts import ChatterboxTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = ChatterboxTTS.from_pretrained(device=device)
    print(f"Model loaded on {device}")
    yield
    del MODEL


app = FastAPI(title="Chatterbox-TTS", lifespan=lifespan)


# ---------- request model ----------
class GenerateRequest(BaseModel):
    text: str
    voice: str = "FR"
    temperature: float | None = None
    exaggeration: float | None = None
    repetition_penalty: float | None = None
    top_k: int | None = None
    cfg_weight: float | None = None


# ---------- health probes ----------
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


# ---------- upload voice ----------
@app.post("/upload-voice")
async def upload_voice(
    file: UploadFile = File(...),
    name: str = Form(...),
):
    """Upload a .wav reference file. The voice name is used as the voice ID."""
    global AVAILABLE_VOICES

    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Only .wav files accepted")

    # Sanitize name — strip extension if accidentally included, reject path traversal
    clean_name = os.path.splitext(name)[0].strip()
    if not clean_name or "/" in clean_name or "\\" in clean_name:
        raise HTTPException(400, "Invalid voice name")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")

    out_path = os.path.join(VOICES_DIR, f"{clean_name}.wav")
    with open(out_path, "wb") as f:
        f.write(contents)

    # Rescan
    AVAILABLE_VOICES = find_voices()
    print(f"Voice '{clean_name}' uploaded ({len(contents)} bytes). Available: {AVAILABLE_VOICES}")

    return {"voice": clean_name, "size": len(contents), "available": AVAILABLE_VOICES}


# ---------- generate ----------
@app.post("/generate")
async def generate(req: GenerateRequest):
    if MODEL is None:
        raise HTTPException(503, "model not loaded")

    if req.voice not in AVAILABLE_VOICES:
        raise HTTPException(
            400,
            f"Unknown voice '{req.voice}'. Available: {', '.join(AVAILABLE_VOICES) or '(none)'}"
        )

    ref_path = get_ref_path(req.voice)
    if not os.path.isfile(ref_path):
        raise HTTPException(404, f"Reference file missing: {ref_path}")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is required")

    gen_kwargs = {}
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.exaggeration is not None:
        gen_kwargs["exaggeration"] = req.exaggeration
    if req.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = req.repetition_penalty
    if req.top_k is not None:
        gen_kwargs["top_k"] = req.top_k
    if req.cfg_weight is not None:
        gen_kwargs["cfg_weight"] = req.cfg_weight

    wav = MODEL.generate(text, audio_prompt_path=ref_path, **gen_kwargs)

    buf = io.BytesIO()
    ta.save(buf, wav, MODEL.sr, format="wav")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")
