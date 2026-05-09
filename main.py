"""Minty Box — integration harness.

Wires wake word detection and speech-to-text together for testing.
Every time a wake word fires, the utterance that follows is recorded
and transcribed, then saved as a timestamped JSON file.

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
    stt = SpeechToText(
        disfluency_filter="confidence",
        custom_words=["Araminta", "Minty", "Moorside", "Conford", "Liphook"],
    )

    # Resolve model paths for the wake word listener.
    model_paths: list[str] | None = None
    if args.model:
        model_paths = []
        for p in args.model:
            path = Path(p).expanduser().resolve()
            if not path.exists():
                parser.error(f"Model not found: {path}")
            model_paths.append(str(path))

    def on_wake(wake_word: str, score: float) -> None:
        """Handle a wake word detection: record, transcribe, save JSON."""
        timestamp = datetime.now(timezone.utc)

        print(
            f"\n🎤 Wake word: '{wake_word}' (score={score:.4f})"
            f"\n   Listening for command..."
        )

        try:
            result = stt.listen_and_transcribe(timeout=10.0)
        except Exception:
            logger.exception("STT failed during listen_and_transcribe")
            return

        # Build the capture record.
        record = {
            "timestamp": timestamp.isoformat(),
            "wake_word": wake_word,
            "wake_score": score,
            "transcription": result.text,
            "raw_transcription": result.raw_text,
            "language": result.language,
            "language_probability": result.language_probability,
            "audio_duration_s": result.duration,
            "filter_applied": result.filter_applied,
        }

        # Write JSON file.
        filename = timestamp.strftime("%Y-%m-%dT%H-%M-%S") + ".json"
        path = output_dir / filename

        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)

        print(f"   Transcription: {result.text!r}")
        print(f"   Saved: {path}\n")

    # Start the wake word listener (blocks).
    listener = WakeWordListener(
        on_wake=on_wake,
        model_paths=model_paths,
        threshold=args.threshold,
        enable_vad=not args.no_vad,
    )
    listener.start()


if __name__ == "__main__":
    main()
