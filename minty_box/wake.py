"""
Wake word detection for the Minty Box.

Listens to the ReSpeaker Lite microphone continuously using openWakeWord
for on-device wake word detection. Runs entirely offline — no network
calls during inference. When a wake word is detected with sufficient
confidence, triggers a user-supplied callback.

Architecture:
    ReSpeaker Lite mic array
        → XMOS DSP (hardware AEC, noise suppression, beamforming)
        → USB audio (16kHz, 16-bit, mono)
        → sounddevice InputStream
        → openWakeWord ONNX model (80ms frames)
        → VAD gating (Silero)
        → threshold + cooldown filtering
        → on_wake callback

Usage (library):
    from minty_box.wake import WakeWordListener

    def on_wake(wake_word: str, score: float) -> None:
        print(f"Wake word '{wake_word}' detected: {score:.3f}")

    listener = WakeWordListener(on_wake=on_wake)
    listener.start()   # blocks until interrupted or stop() called

Usage (CLI):
    python -m minty_box.wake --model models/araminta.onnx
    python -m minty_box.wake --threshold 0.7 --debug
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Substring used to identify the ReSpeaker Lite in sounddevice queries.
# Matches: "ReSpeaker Lite: USB Audio (hw:CARD=Lite,DEV=0)"
_RESPEAKER_NAME: str = "ReSpeaker Lite"

# Audio format required by openWakeWord's ONNX models.
# 16 kHz sample rate, mono, 16-bit signed integer PCM.
_SAMPLE_RATE: int = 16000
_CHANNELS: int = 1
_BLOCK_SIZE: int = 1280  # 80 ms at 16 kHz — openWakeWord's native frame size
_DTYPE: str = "int16"

# Pre-trained models bundled with the openWakeWord package.
_BUILTIN_MODELS: list[str] = [
    "alexa",
    "hey_mycroft",
    "hey_jarvis",
    "timer",
    "weather",
]

# Defaults for detection behaviour.
_DEFAULT_THRESHOLD: float = 0.5
_COOLDOWN_SECONDS: float = 2.0
_VAD_THRESHOLD: float = 0.5  # Silero VAD probability threshold


# ---------------------------------------------------------------------------
# WakeWordListener
# ---------------------------------------------------------------------------

class WakeWordListener:
    """Continuous wake word detector using openWakeWord and sounddevice.

    Opens an audio input stream from the ReSpeaker Lite (or a
    user-specified device) and feeds 80 ms frames into one or more
    openWakeWord ONNX models.  Predictions are gated by a configurable
    score threshold and an optional Silero voice-activity detector.
    Repeated detections of the same wake word within a cooldown window
    are suppressed.  When a valid detection occurs, the user-supplied
    ``on_wake`` callback is invoked.

    Parameters
    ----------
    on_wake:
        Callback invoked for each detection that passes threshold,
        VAD, and cooldown filters.  Receives the model name and the
        raw prediction score (0–1).
    model_paths:
        Paths to custom ``.onnx`` models.  If ``None``, all built-in
        models (``alexa``, ``hey_mycroft``, …) are loaded.
    threshold:
        Minimum score (0–1) required to consider a frame a detection.
        Lower values increase sensitivity but may raise false positives.
    input_device:
        sounddevice device identifier (integer index or name substring).
        If ``None``, the method searches for a device whose name
        contains ``"ReSpeaker Lite"``.
    enable_vad:
        When ``True`` (default), raw predictions are multiplied by the
        Silero VAD probability for the same frame.  Frames where VAD
        probability is below ``_VAD_THRESHOLD`` (0.5) are effectively
        suppressed.  This significantly reduces false activations from
        continuous background noise.
    cooldown:
        Minimum interval (seconds) between successive detections of the
        *same* model.  Different models may fire independently.

    Raises
    ------
    RuntimeError
        If openWakeWord model loading fails (e.g., corrupted ``.onnx``
        file or incompatible ONNX Runtime version).

    Notes
    -----
    - SpeexDSP noise suppression is not available on ARM64 (Raspberry
      Pi).  The XMOS DSP on the ReSpeaker Lite provides hardware noise
      suppression instead.
    - The ONNX Runtime CUDA provider is not available on the Pi; the
      warning printed during model load is harmless.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

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
        """Initialise the listener.

        See the class docstring for parameter descriptions.
        """
        self._on_wake = on_wake
        self._threshold = threshold
        self._cooldown = cooldown

        # Resolve audio input device.
        self._device = input_device or self._find_respeaker()
        logger.info("Using input device: %s", self._device)

        # SpeexDSP is x86-only.  On ARM64 the XMOS DSP handles noise
        # suppression in hardware, so we disable it gracefully.
        use_speex: bool = self._speex_available()

        # Load wake word model(s).
        if model_paths is None:
            logger.info("Loading all built-in models: %s", _BUILTIN_MODELS)
            self._model = Model(
                enable_speex_noise_suppression=use_speex,
                vad_threshold=_VAD_THRESHOLD if enable_vad else 0,
            )
            self._model_names: list[str] = _BUILTIN_MODELS
        else:
            logger.info("Loading custom models: %s", model_paths)
            self._model = Model(
                wakeword_model_paths=model_paths,
                enable_speex_noise_suppression=use_speex,
                vad_threshold=_VAD_THRESHOLD if enable_vad else 0,
            )
            self._model_names = [Path(p).stem for p in model_paths]

        # Per-model cooldown timestamps.
        self._last_detection: dict[str, float] = {}

        # Runtime state.
        self._stream: Optional[sd.InputStream] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the audio stream and block until stopped.

        Installs signal handlers for ``SIGINT`` and ``SIGTERM`` so
        the process can be cleanly interrupted.  Returns when
        :meth:`stop` is called (from another thread) or a signal
        is received.

        Raises
        ------
        OSError
            If the audio device cannot be opened (e.g., device busy,
            insufficient permissions, or device disappeared).
        """
        self._running = True

        # Graceful shutdown on Ctrl+C / kill.
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
        """Request graceful shutdown.

        Thread-safe.  May be called from any thread, including a
        callback invoked by :meth:`start`.  The listener will exit
        at the next iteration of the main loop.
        """
        self._running = False

    # ------------------------------------------------------------------
    # Audio callback (executed by sounddevice on its background thread)
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        timestamp: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Process one audio block from the input stream.

        Called by sounddevice on its internal background thread for
        every ``_BLOCK_SIZE`` samples (80 ms at 16 kHz).  The audio is
        fed to openWakeWord and predictions are checked against the
        configured threshold, VAD gate, and cooldown window.

        Parameters
        ----------
        indata:
            Audio samples of shape ``(frames, channels)`` with dtype
            ``int16``.  For mono input this will be ``(1280, 1)``.
        frames:
            Number of sample frames in this block (always ``_BLOCK_SIZE``).
        timestamp:
            CData structure with ADC/DAC time and current stream time.
        status:
            Bitfield of :class:`sounddevice.CallbackFlags`.  Non-zero
            indicates an over/underflow or abort condition.

        Warns
        -----
        Logs a warning if ``status`` is non-zero (stream glitch).
        """
        if status:
            logger.warning("Audio stream status: %s", status)

        # Collapse to 1-D int16 — openWakeWord requirement.
        audio: np.ndarray = indata[:, 0] if indata.ndim > 1 else indata

        predictions: dict[str, float] = self._model.predict(audio)
        now: float = time.monotonic()

        for model_name, score in predictions.items():
            if score < self._threshold:
                continue

            # Cooldown: suppress repeated triggers of the same model.
            last: float = self._last_detection.get(model_name, 0.0)
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
                logger.exception(
                    "Unhandled exception in on_wake callback for model '%s'",
                    model_name,
                )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _speex_available() -> bool:
        """Return ``True`` if the SpeexDSP Python wheel is installed.

        SpeexDSP provides a lightweight software noise suppressor.
        The upstream wheel (``speexdsp_ns``) is compiled for x86-64
        only and will not install on ARM64.  On the Pi we rely on
        the XMOS DSP hardware instead.

        Returns
        -------
        bool
            ``True`` if ``speexdsp_ns`` can be imported.
        """
        try:
            from speexdsp_ns import NoiseSuppression  # noqa: F401
            return True
        except ModuleNotFoundError:
            logger.debug(
                "SpeexDSP not available — XMOS DSP handles "
                "noise suppression in hardware"
            )
            return False

    @staticmethod
    def _find_respeaker() -> str | int:
        """Locate the ReSpeaker Lite in sounddevice's device list.

        Iterates all audio devices and returns the index of the first
        whose ``name`` field contains ``"ReSpeaker Lite"``.  Using the
        integer index avoids ambiguity when multiple devices share
        substring matches.

        Returns
        -------
        str | int
            sounddevice device identifier (integer index if found,
            system default input device otherwise).

        Warns
        -----
        Logs a warning and falls back to the system default if no
        ReSpeaker Lite is detected.
        """
        devices: list[dict] = sd.query_devices()
        for idx, dev in enumerate(devices):
            if _RESPEAKER_NAME in dev["name"]:
                return idx
        logger.warning(
            "ReSpeaker Lite not found in devices. "
            "Using system default. Available devices: %s",
            [d["name"] for d in devices],
        )
        return sd.default.device[0]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _handle_signal(self, signum: int, frame: object) -> None:
        """Signal handler for SIGINT / SIGTERM.

        Sets ``_running`` to ``False``, causing :meth:`start` to exit
        its main loop at the next 100 ms check.

        Parameters
        ----------
        signum:
            Signal number received.
        frame:
            Current stack frame (unused).
        """
        logger.info("Received signal %d; shutting down...", signum)
        self._running = False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the wake word listener from the command line.

    Parses command-line arguments and starts a :class:`WakeWordListener`
    with the user's configuration.  Detections are printed to stdout.

    Exit codes:
        0 — clean shutdown (interrupt or signal).
        2 — invalid arguments (handled by argparse).
    """
    parser = argparse.ArgumentParser(
        description="Listen for wake words using openWakeWord",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        help=(
            "Path(s) to custom .onnx model(s). "
            "Omit to use all built-in models (%s)."
        ) % ", ".join(_BUILTIN_MODELS),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        help=(
            "Detection threshold (0–1). "
            "Higher values reduce false positives. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "sounddevice input device (name or index). "
            "Auto-detects ReSpeaker Lite if omitted."
        ),
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD gating (may increase false activations).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all non-detection log output.",
    )

    args = parser.parse_args()

    # Configure logging.
    if args.debug:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    def on_wake(wake_word: str, score: float) -> None:
        """Print detection to stdout — always visible regardless of log level."""
        print(f"\n*** WAKE WORD: {wake_word} (score={score:.4f}) ***\n")

    # Resolve model paths, validating existence.
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
