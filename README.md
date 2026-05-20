# Minty Box

Araminta's physical hardware companion — a Raspberry Pi 5 with ReSpeaker Lite
voice interface, speaker, and touch display.  A complete wake-word → STT →
agent → TTS pipeline that routes voice queries to the real Hermes agent
(the same one that answers on Discord and Telegram).

## Hardware

- Raspberry Pi 5, 8GB
- NVMe Base + 1TB SSD
- ReSpeaker Lite (XIAO ESP32S3) — mic array + XMOS DSP
- Mono Enclosed Speaker 4R 5W
- 3.4" HDMI Round Touch Display 800×800
- Custom 3D printed case

Parts ordered from Pi Hut, May 2026.

## Architecture (two-gateway)

```
ReSpeaker mics → XMOS DSP → USB audio → Pi (ALSA)
    → openWakeWord ("Araminta" wake word, VAD-gated)
        → on_wake: confirmation beep → utterance capture (Silero VAD)
            → faster-whisper STT
                → HermesAPIHandler → POST :8643/v1/chat/completions
                    → gemma4-voice (the real Araminta, with SOUL.md + memory)
                        → Kokoro TTS (:8880, voice bf_emma)
                            → ReSpeaker speaker
```

Target end-to-end latency: ~10 seconds.

## Repository

Public, default branch `master`.  Python 3.14.3 via pyenv, managed with UV.

## Project Structure

```
minty-box/
  minty_box/
    __init__.py     # Package root, exports WakeWordListener
    wake.py         # Wake word listener (openWakeWord + VAD + capture)
    stt.py          # Speech-to-text (faster-whisper)
    handler.py      # Hermes API handler (routes to :8643 agent)
    tts.py          # Kokoro TTS client
    speaker.py      # WAV playback via ffplay / ALSA
    cli.py          # minty-box CLI (start / stop / status)
  main.py           # Standalone listener entry point
  tests/            # pytest suite (100 tests)
  models/           # .onnx wake word models (gitignored, add manually)
  captures/         # Runtime JSON logs (gitignored)
  docs/
    respeaker-lite-setup.md
    wake-word-training.md
  pyproject.toml    # UV project config
```

## Usage

```bash
# Install dependencies
uv sync

# Run the full voice pipeline (foreground)
uv run python main.py --model models/Araminta.onnx --threshold 0.3

# Or use the CLI launcher (background subprocess, recommended)
minty-box start --model models/Araminta.onnx --threshold 0.3
minty-box status
minty-box stop

# Run tests
uv run pytest          # full suite (100 tests)
uv run pytest -x       # stop on first failure
```

## Docs

- [ReSpeaker Lite Setup](docs/respeaker-lite-setup.md) — firmware flashing,
  ALSA configuration, power requirements, PulseAudio sink setup, volume
  calibration, cable pitfalls, and diagnostic checklists.  Follow this
  **exactly**.
- [Wake Word Training](docs/wake-word-training.md) — train a custom
  "Araminta" wake word model using openWakeWord's Colab notebook.

## Dependencies

Runtime: `openwakeword`, `onnxruntime`, `faster-whisper`, `sounddevice`,
`soundfile`, `numpy`, `kokoro` (TTS server runs separately on :8880).

The Hermes agent gateway must be running locally on `:8643` — the voice
pipeline is only the **client**; the agent intelligence lives there.
