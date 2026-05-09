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

## Docs

- [ReSpeaker Lite Setup Guide](docs/respeaker-lite-setup.md) — complete
  step-by-step installation, covering firmware flashing, ALSA config, power,
  PulseAudio, volume calibration, cable pitfalls, and diagnostic checklists.
  Follow this **exactly**.

## Project Setup

```bash
git clone git@github.com:djmcnay/minty-box.git
cd minty-box
# pyenv will auto-switch to 3.14.3 if inited
uv sync
```
