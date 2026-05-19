Nightly TTS batch processing on SaladCloud GPU instances.

## Voices

Voice reference `.wav` files are **not baked into the image**. Upload them at runtime:

```bash
curl -X POST https://<gateway>/upload-voice \
  -F "name=FR" \
  -F "file=@FR.wav"
```

The filename (minus `.wav`) becomes the voice ID. All voices uploaded at runtime are ephemeral — re-upload on each container start.

## API

### GET /voices
List available voices.

### POST /upload-voice
Upload a `.wav` reference file. Multipart form: `name` (voice ID) + `file` (.wav).

### POST /generate
```json
{"text": "Hello world", "voice": "FR"}
```
Returns WAV audio.

Optional params: `temperature`, `exaggeration`, `repetition_penalty`, `top_k`, `cfg_weight` — all nullable, model defaults apply when omitted.

### Health probes (SaladCloud required)
- GET /started
- GET /live
- GET /ready

## Deploy

Push to `main` → GitHub Actions builds and pushes to `ghcr.io/paulc007/chatterbox-salad`.
SaladCloud pulls from there via the Container Engine API.
