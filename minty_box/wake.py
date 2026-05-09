"""
Wake word detection for the Minty Box.

Listens to the ReSpeaker Lite microphone continuously using openWakeWord
for on-device wake word detection. Runs entirely offline. When a wake word
is detected with sufficient confidence, triggers the configured callback.

Usage:
    from minty_box.wake import WakeWordListener

    def on_wake(wake_word: str, score: float):
        print(f"Wake word '{wake_word}' detected: {score:.3f}")

    listener = WakeWordListener(on_wake=on_wake)
    listener.start()   # blocks until interrupted

Or as a standalone script:
    python -m minty_box.wake --model araminta.onnx
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# ALSA device identifier for the ReSpeaker Lite
# sounddevice sees it as: "ReSpeaker Lite: USB Audio (hw:CARD=Lite,DEV=0)"
# We match by substring so it survives index changes.
_RESPEAKER_NAME = "ReSpeaker Lite"

# Audio format openWakeWord expects
_SAMPLE_RATE = 16000
_CHANNELS = 1
_BLOCK_SIZE = 1280  # 80ms at 16kHz — openWakeWord's native frame size
_DTYPE = "int16"

# Built-in models that come with openWakeWord
_BUILTIN_MODELS: list[str] = [
    "alexa",
    "hey_mycroft",
    "hey_jarvis",
    "timer",
    "weather",
]

# Default wake word threshold
_DEFAULT_THRESHOLD = 0.5

# How long to suppress repeated detections of the same wake word (seconds)
_COOLDOWN_SECONDS = 2.0


# ---------------------------------------------------------------------------
# WakeWordListener
# ---------------------------------------------------------------------------

class WakeWordListener:
    """
    Continuous wake word detector using openWakeWord and sounddevice.

    Parameters
    ----------
    on_wake : callable(wake_word: str, score: float) -> None
        Called when a wake word is detected with confidence above threshold.
    model_paths : list[str] | None
        Paths to custom .onnx models. If None, loads all built-in models.
    threshold : float
        Score threshold (0–1). Default 0.5.
    input_device : str | int | None
        sounddevice device identifier. If None, auto-detects ReSpeaker Lite.
    enable_vad : bool
        Use Silero VAD to gate predictions. Reduces false activations from
        background noise. Default True for quiet cottage environments.
    cooldown : float
        Seconds to suppress repeated detections of the same wake word.
    """

    def __init__(
        self,
        on_wake: Callable[[str, float], None],
        *,
        model_paths: list[str] | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
        input_device: str | int | None = None,
        enable_vad: bool = True,
        cooldown: float = _COOLDOWN_SECONDS,
    ) -> None:
        self._on_wake = on_wake
        self._threshold = threshold
        self._cooldown = cooldown

        # Resolve input device
        self._device = input_device or self._find_respeaker()
        logger.info("Using input device: %s", self._device)

        # Speex noise suppression requires an x86-only wheel. On ARM64
        # (Raspberry Pi), the ReSpeaker's XMOS DSP handles noise suppression
        # in hardware anyway, so we disable it gracefully.
        use_speex = self._speex_available()

        # Resolve model paths
        if model_paths is None:
            # Use built-in models
            logger.info("Loading all built-in models: %s", _BUILTIN_MODELS)
            self._model = Model(
                enable_speex_noise_suppression=use_speex,
                vad_threshold=0.5 if enable_vad else 0,
            )
            self._model_names = _BUILTIN_MODELS
        else:
            # Custom model(s) only
            logger.info("Loading custom models: %s", model_paths)
            self._model = Model(
                wakeword_model_paths=model_paths,
                enable_speex_noise_suppression=use_speex,
                vad_threshold=0.5 if enable_vad else 0,
            )
            self._model_names = [
                Path(p).stem for p in model_paths
            ]

        # Cooldown tracking: {model_name: timestamp_of_last_detection}
        self._last_detection: dict[str, float] = {}

        # Stream handle
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the wake word listener. Blocks until interrupted (Ctrl+C
        or SIGTERM) or until stop() is called from another thread.
        """
        self._running = True

        # Set up graceful shutdown on SIGINT/SIGTERM
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(
            "Listening for wake words: %s (threshold=%.2f, device=%s)",
            self._model_names,
            self._threshold,
            self._device,
        )

        try:
            with sd.InputStream(
                device=self._device,
                samplerate=_SAMPLE_RATE,
                channels=_CHANNELS,
                dtype=_DTYPE,
                blocksize=_BLOCK_SIZE,
                callback=self._audio_callback,
            ):
                logger.debug("Audio stream opened; waiting for wake words...")
                while self._running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._running = False
            logger.info("Listener stopped")

    def stop(self) -> None:
        """Stop the listener (can be called from any thread)."""
        self._running = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        timestamp: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice for each audio block."""
        if status:
            logger.warning("Audio stream status: %s", status)

        # openWakeWord expects 1D int16 array
        audio = indata[:, 0] if indata.ndim > 1 else indata

        # Get predictions (dict of model_name -> score)
        predictions = self._model.predict(audio)

        now = time.monotonic()

        for model_name, score in predictions.items():
            if score < self._threshold:
                continue

            # Cooldown: don't fire repeatedly for the same wake word
            last = self._last_detection.get(model_name, 0)
            if now - last < self._cooldown:
                continue
            self._last_detection[model_name] = now

            logger.info(
                "Wake word DETECTED: %s (score=%.4f)",
                model_name,
                score,
            )
            try:
                self._on_wake(model_name, float(score))
            except Exception:
                logger.exception("on_wake callback raised")

    @staticmethod
    def _speex_available() -> bool:
        """Check if SpeexDSP noise suppression is available (x86 only)."""
        try:
            from speexdsp_ns import NoiseSuppression  # noqa: F401
            return True
        except ModuleNotFoundError:
            logger.debug("SpeexDSP not available — XMOS DSP handles noise suppression in hardware")
            return False

    @staticmethod
    def _find_respeaker() -> str:
        """Find the ReSpeaker Lite device identifier for sounddevice."""
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if _RESPEAKER_NAME in dev["name"]:
                # Use device index for reliable selection
                return idx  # type: ignore[return-value]
        # Fallback: try the name as a substring match
        logger.warning(
            "ReSpeaker Lite not found in devices. "
            "Using system default. Available devices: %s",
            [d["name"] for d in devices],
        )
        return sd.default.device[0]  # type: ignore[return-value]

    def _handle_signal(self, signum: int, frame: object) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info("Received signal %d; shutting down...", signum)
        self._running = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Standalone wake word listener with configurable model and threshold."""
    parser = argparse.ArgumentParser(
        description="Listen for wake words using openWakeWord",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        help="Path(s) to custom .onnx model(s). Omit to use built-in models.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        help=f"Detection threshold (0–1, default: {_DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="sounddevice input device (name or index). Auto-detects ReSpeaker if omitted.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD gating (may increase false activations)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print detection events",
    )

    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.debug else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    def on_wake(wake_word: str, score: float) -> None:
        # Always print detections regardless of log level
        print(f"\n*** WAKE WORD: {wake_word} (score={score:.4f}) ***\n")

    model_paths: list[str] | None = None
    if args.model:
        model_paths = []
        for p in args.model:
            path = Path(p).expanduser().resolve()
            if not path.exists():
                parser.error(f"Model not found: {path}")
            model_paths.append(str(path))

    listener = WakeWordListener(
        on_wake=on_wake,
        model_paths=model_paths,
        threshold=args.threshold,
        input_device=args.device,
        enable_vad=not args.no_vad,
    )
    listener.start()


if __name__ == "__main__":
    main()
