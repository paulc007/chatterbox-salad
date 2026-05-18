# Chatterbox Salad

Nightly TTS batch processing on SaladCloud GPU instances.

## Voices

Drop `.wav` files in `voices/` — the filename (minus `.wav`) becomes the voice ID.

Current voices:
- AF, AH, DG, DT, FR

## API

### `GET /voices`
List available voices.

### `POST /generate`
```json
{"text": "Hello world", "voice": "FR"}
```
Returns WAV audio.

### Health probes (SaladCloud required)
- `GET /started`
- `GET /live`
- `GET /ready`

## Deploy

GitHub Actions builds and pushes to `ghcr.io/paulc007/chatterbox-salad`.
SaladCloud pulls from there via the Container Engine API.
