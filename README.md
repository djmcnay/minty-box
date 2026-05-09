# Minty Box

Araminta's physical hardware companion — a Raspberry Pi 5 with ReSpeaker Lite
voice interface, speaker, and touch display.

## Hardware

- Raspberry Pi 5, 8GB
- NVMe Base + 1TB SSD
- ReSpeaker Lite (XIAO ESP32S3) — mic array + DSP
- Mono Enclosed Speaker 4R 5W
- 3.4" HDMI Round Touch Display 800x800
- Custom 3D printed case

Parts ordered from Pi Hut, May 2026.

## Repository

Public, default branch `master`.

Python 3.14.3 via pyenv, managed with UV.

## Project Structure

```
minty-box/
  minty_box/
    __init__.py     # Package root, exports WakeWordListener
    wake.py         # Wake word listener using openWakeWord
  models/           # Custom .onnx wake word models (gitignored, add manually)
  docs/             # Setup guides and documentation
    respeaker-lite-setup.md
    wake-word-training.md
  pyproject.toml    # UV project config and dependencies
  main.py           # Placeholder — eventual entry point
```

## Pipeline

```
ReSpeaker mics → XMOS DSP → USB audio → Pi (ALSA)
    → wake.py (openWakeWord, VAD-gated) → trigger → faster-whisper (STT)
    → Hermes → Kokoro (TTS) → ReSpeaker speaker
```

### Wake Word Detection

```bash
# With built-in models (Alexa, Hey Mycroft, Hey Jarvis, etc.)
uv run python -m minty_box.wake

# With a custom model (e.g., "Araminta")
uv run python -m minty_box.wake --model models/araminta.onnx

# Tune threshold and device
uv run python -m minty_box.wake --model models/araminta.onnx --threshold 0.7
```

See [docs/wake-word-training.md](docs/wake-word-training.md) for training a
custom "Araminta" wake word model using Google Colab.

## Docs

- [ReSpeaker Lite Setup Guide](docs/respeaker-lite-setup.md) — complete
  step-by-step installation, covering firmware flashing, ALSA config, power,
  PulseAudio, volume calibration, cable pitfalls, and diagnostic checklists.
  Follow this **exactly**.
- [Wake Word Training](docs/wake-word-training.md) — train a custom
  "Araminta" wake word model using openWakeWord's Colab notebook.

## Project Setup

```bash
git clone git@github.com:djmcnay/minty-box.git
cd minty-box
# pyenv will auto-switch to 3.14.3 if inited
uv sync
```
