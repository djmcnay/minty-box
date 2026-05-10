"""Minty Box — integration harness.

Wires wake word detection and speech-to-text together for testing.
Every time a wake word fires, the utterance that follows is captured
from the same audio stream and transcribed, then saved as a
timestamped JSON file.

Architecture:
    Single ALSA stream → wake word detection (openWakeWord)
        → on_wake fires → capture mode (same stream, RMS silence detection)
        → on_utterance delivers buffer → stt.transcribe_buffer()
        → JSON saved to captures/

Usage:
    uv run python main.py                     # all built-in models
    uv run python main.py --model models/araminta.onnx   # custom model
    uv run python main.py --output-dir /tmp/captures      # custom output path

Output (per detection):
    captures/2026-05-09T22-45-30.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from minty_box.stt import SpeechToText
from minty_box.wake import WakeWordListener

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the integration harness — wake word → STT → JSON."""
    parser = argparse.ArgumentParser(
        description="Minty Box — wake word + STT integration test harness",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        help="Path(s) to custom .onnx model(s). Omit to use built-in models (includes hey_jarvis).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Wake word detection threshold (0–1, default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="captures",
        help="Directory for JSON capture files (default: captures/)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD gating for wake word detection",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Logging.
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Output directory.
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Captures will be saved to: %s", output_dir)

    # Build the STT engine (shared instance — loads model once).
    # base.en: 74M params, ~6x real-time on Pi 5 CPU — much better
    # accuracy than tiny.en while still well within real-time budgets.
    # large-v3-turbo would be more accurate but ~1.2x real-time on Pi 5
    # (a 5s utterance takes 20-25s to transcribe — unacceptable for
    # voice-assistant use).
    stt = SpeechToText(
        model_size="base.en",
        disfluency_filter=None,  # disabled — logprobs unreliable on arm64 int8
        custom_words=["Araminta", "Minty", "Moorside", "Conford", "Liphook"],
    )

    # Resolve model paths for the wake word listener.
    # Default: only hey_jarvis (until custom "Araminta" model is trained).
    model_paths: list[str] | None = None
    if args.model:
        model_paths = []
        for p in args.model:
            path = Path(p).expanduser().resolve()
            if not path.exists():
                parser.error(f"Model not found: {path}")
            model_paths.append(str(path))
    else:
        # Single built-in model extracted to models/ for portability.
        model_paths = [str(Path(__file__).parent / "models" / "hey_jarvis_v0.1.onnx")]

    # Track the last wake word details for the JSON record.
    last_wake: dict = {}

    def on_wake(wake_word: str, score: float) -> None:
        """Called when a wake word is detected.  Stores details for the
        subsequent on_utterance callback."""
        last_wake["word"] = wake_word
        last_wake["score"] = score
        last_wake["timestamp"] = datetime.now(timezone.utc)

        print(
            f"\n🎤 Wake word: '{wake_word}' (score={score:.4f})"
            f"\n   Listening for command..."
        )

    def on_utterance(audio: np.ndarray) -> None:
        """Called with the captured utterance buffer (int16, 16kHz mono).
        Transcribes and saves a JSON record."""
        wake_timestamp: datetime | None = last_wake.pop("timestamp", None)
        if wake_timestamp is None:
            logger.warning("on_utterance called without a preceding wake word")
            return

        wake_word: str = last_wake.pop("word", "unknown")
        wake_score: float = last_wake.pop("score", 0.0)

        try:
            result = stt.transcribe_buffer(audio)
        except Exception:
            logger.exception("STT failed during transcribe_buffer")
            return

        # Build the capture record.
        record = {
            "timestamp": wake_timestamp.isoformat(),
            "wake_word": wake_word,
            "wake_score": wake_score,
            "transcription": result.text,
            "raw_transcription": result.raw_text,
            "language": result.language,
            "language_probability": result.language_probability,
            "audio_duration_s": result.duration,
            "filter_applied": result.filter_applied,
        }

        # Write JSON file.
        filename = wake_timestamp.strftime("%Y-%m-%dT%H-%M-%S") + ".json"
        path = output_dir / filename

        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)

        print(f"   Transcription: {result.text!r}")
        print(f"   Saved: {path}\n")

    # Start the wake word listener (blocks).
    listener = WakeWordListener(
        on_wake=on_wake,
        on_utterance=on_utterance,
        model_paths=model_paths,
        threshold=args.threshold,
        enable_vad=not args.no_vad,
    )
    listener.start()


if __name__ == "__main__":
    main()
