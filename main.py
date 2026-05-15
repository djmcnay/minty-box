"""Minty Box — integration harness.

Wires wake word detection, speech-to-text, Direct LLM query, text-to-speech,
and audio playback together as a complete voice-assistant pipeline.

Architecture:
    Single ALSA stream → wake word detection (openWakeWord)
        → on_wake fires → beep → capture mode (same stream, silence detection)
        → on_utterance delivers buffer → stt.transcribe_buffer()
        → handler.process(text) → Direct LLM API (fast cloud model)
        → _clean_for_speech(response)
        → tts.synthesize(text) → Kokoro HTTP API (local)
        → speaker.play(wav) → ReSpeaker Lite speaker

Target end-to-end latency: ~10 seconds.

Usage:
    uv run python main.py                                    # defaults
    uv run python main.py --model models/Araminta.onnx       # custom wake word
    uv run python main.py --threshold 0.55                   # detection threshold
    uv run python main.py --llm-model gemini-3-flash-preview:latest
    uv run python main.py --no-tts                           # text-only
    uv run python main.py --debug                            # verbose logging
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError

import numpy as np

from minty_box.handler import DirectLLMHandler, HermesAPIHandler
from minty_box.speaker import Speaker
from minty_box.stt import SpeechToText
from minty_box.tts import KokoroTTS
from minty_box.handler import VOICE_MODEL as DEFAULT_LLM_MODEL
from minty_box.wake import WakeWordListener

logger = logging.getLogger(__name__)

# ── global instances (lazy-init) ───────────────────────────────────────
_stt: SpeechToText | None = None
_tts: KokoroTTS | None = None
_speaker: Speaker | None = None


def _get_stt() -> SpeechToText:
    global _stt
    if _stt is None:
        _stt = SpeechToText(
            model_size="tiny.en",
            disfluency_filter=None,
            custom_words=[
                "Araminta", "Minty", "Moorside", "Conford", "Liphook",
            ],
        )
    return _stt


def _get_tts() -> KokoroTTS:
    global _tts
    if _tts is None:
        _tts = KokoroTTS()
    return _tts


def _get_speaker() -> Speaker:
    global _speaker
    if _speaker is None:
        _speaker = Speaker()
    return _speaker


# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minty Box — voice assistant integration harness",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        help="Path(s) to custom .onnx wake word model(s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Wake word detection threshold (0–1, default: 0.55)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=(
            "LLM model tag for voice responses "
            f"(default: {DEFAULT_LLM_MODEL})"
        ),
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech — print responses to stdout only.",
    )
    parser.add_argument(
        "--no-captures",
        action="store_true",
        help="Don't save capture JSONs.",
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

    # ── output directory ───────────────────────────────────────────────
    captures_enabled = not args.no_captures
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Captures will be saved to: %s", output_dir)

    # ── STT engine ─────────────────────────────────────────────────────
    stt = _get_stt()

    # ── TTS warm-up (eager, so we fail early if Kokoro is down) ───────
    if not args.no_tts:
        try:
            tts = _get_tts()
            tts.synthesize(".")  # one-character warm-up
            logger.info("Kokoro TTS ready.")
        except (URLError, OSError, ValueError) as e:
            logger.warning(
                "Kokoro TTS unavailable (%s).  Falling back to --no-tts.", e
            )
            args.no_tts = True

    speaker = _get_speaker() if not args.no_tts else None

    # ── handler: Hermes API (the real agent) ────────────────────────────
    handler = HermesAPIHandler()
    logger.info(
        "Using Hermes API handler (agent=%s, timeout=%ds).",
        "hermes-agent",
        handler._timeout,
    )

    # ── resolve wake word model paths ─────────────────────────────────
    model_paths: list[str] | None = None
    if args.model:
        model_paths = []
        for p in args.model:
            path = Path(p).expanduser().resolve()
            if not path.exists():
                parser.error(f"Model not found: {path}")
            model_paths.append(str(path))
    else:
        model_paths = [
            str(Path(__file__).parent / "models" / "hey_jarvis_v0.1.onnx")
        ]

    # ── callbacks ──────────────────────────────────────────────────────
    last_wake: dict = {}

    def on_wake(wake_word: str, score: float) -> None:
        last_wake["word"] = wake_word
        last_wake["score"] = score
        last_wake["timestamp"] = datetime.now(timezone.utc)
        if not args.no_tts and speaker is not None:
            try:
                speaker.beep()
            except Exception:
                pass  # beep is cosmetic — never block the pipeline
        print(
            f"\n🎤 Wake word: '{wake_word}' ({score:.4f})"
            f"\n   Listening for command..."
        )

    def on_utterance(audio: np.ndarray) -> None:
        wake_timestamp: datetime | None = last_wake.pop("timestamp", None)
        if wake_timestamp is None:
            logger.warning(
                "on_utterance called without a preceding wake word"
            )
            return

        wake_word: str = last_wake.pop("word", "unknown")
        wake_score: float = last_wake.pop("score", 0.0)

        # 1. Transcribe.
        try:
            result = stt.transcribe_buffer(audio)
        except Exception:
            logger.exception("STT failed")
            return

        transcription = result.text.strip()
        print(f"   Transcription: {transcription!r}")

        # 2. Save capture JSON.
        if captures_enabled:
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
                "handler": "hermes-api",
            }
            filename = (
                wake_timestamp.strftime("%Y-%m-%dT%H-%M-%S") + ".json"
            )
            path = output_dir / filename
            with open(path, "w") as f:
                json.dump(record, f, indent=2, default=str)
            print(f"   Saved: {path}")

        if not transcription:
            return

        # 3. Route to Direct LLM handler.
        try:
            response = handler.process(transcription)
        except Exception:
            logger.exception("Handler failed")
            response = "Sorry, I couldn't process that."

        print(f"   Response: {response!r}")

        # 4. TTS → speaker.
        if not args.no_tts and speaker is not None:
            try:
                tts = _get_tts()
                wav = tts.synthesize(response)
                speaker.play(wav)
            except URLError:
                logger.warning("TTS unavailable — response text-only")
            except Exception:
                logger.exception("TTS/playback failed")

    # ── start listener (blocks) ────────────────────────────────────────
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
