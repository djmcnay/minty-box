"""
Wake word detection and utterance capture for the Minty Box.

Listens to the ReSpeaker Lite microphone continuously using openWakeWord
for on-device wake word detection. Runs entirely offline. When a wake word
is detected, the listener switches into capture mode — accumulating audio
frames from the same stream, monitoring RMS energy for silence, then
emitting the full utterance buffer. Never opens a second ALSA stream.

Architecture:
    ReSpeaker Lite mic array
        → XMOS DSP (hardware AEC, noise suppression, beamforming)
        → USB audio (16kHz, 16-bit, mono)
        → sounddevice InputStream (single stream, shared)
        → openWakeWord ONNX model (80ms frames, listening mode)
        → VAD gating (Silero)
        → threshold + cooldown filtering
        → on_wake callback
        → capture mode (same stream, RMS silence detection)
        → on_utterance callback (raw audio buffer)

Usage:
    from minty_box.wake import WakeWordListener

    def on_wake(wake_word: str, score: float) -> None:
        print(f"Wake word: {wake_word}")

    def on_utterance(audio: np.ndarray) -> None:
        # Send audio to STT, e.g. stt.transcribe_buffer(audio)

    listener = WakeWordListener(on_wake=on_wake, on_utterance=on_utterance)
    listener.start()   # blocks until interrupted
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from collections import deque
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
_RESPEAKER_NAME: str = "ReSpeaker Lite"

# Audio format required by openWakeWord's ONNX models.
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

# Utterance capture defaults.
_CAPTURE_SILENCE_DURATION: float = 1.5  # seconds of quiet before endpoint
_CAPTURE_SILENCE_THRESHOLD: float = 0.01  # RMS amplitude below which is silence
_CAPTURE_TIMEOUT: float = 15.0  # max capture duration
_CAPTURE_MIN_SPEECH: float = 0.5  # minimum speech before allowing endpoint


# ---------------------------------------------------------------------------
# WakeWordListener
# ---------------------------------------------------------------------------

class WakeWordListener:
    """Continuous wake word detector with built-in utterance capture.

    Opens a single audio input stream from the ReSpeaker Lite.  In
    *listening mode*, 80 ms frames are fed into openWakeWord ONNX
    models.  When a wake word fires, the listener switches to *capture
    mode* on the same stream — accumulating frames, monitoring RMS
    energy, detecting silence, then emitting the utterance buffer via
    ``on_utterance``.  After capture, it returns to listening mode.

    This single-stream architecture avoids ALSA ``Device unavailable``
    errors that occur when two capture streams compete for the same
    hardware.

    Parameters
    ----------
    on_wake:
        Callback invoked when a wake word is detected.  Receives the
        model name and the raw prediction score (0–1).
    on_utterance:
        Callback invoked after capture completes, receiving the full
        utterance audio buffer as a 1-D ``int16`` numpy array.  Called
        from the sounddevice background thread — ensure thread-safety
        if interacting with shared state.
    model_paths:
        Paths to custom ``.onnx`` models.  If ``None``, all built-in
        models are loaded.
    threshold:
        Minimum score (0–1) required to trigger a detection.
    input_device:
        sounddevice device identifier.  Auto-detects ReSpeaker Lite
        if ``None``.
    enable_vad:
        When ``True``, Silero VAD gates predictions (reduces false
        activations from background noise).
    cooldown:
        Minimum interval (seconds) between detections of the same model.
    capture_silence_duration:
        Seconds of continuous silence before ending an utterance capture.
    capture_silence_threshold:
        RMS amplitude below which audio is considered silence.
    capture_timeout:
        Maximum capture duration in seconds.  If reached, the
        accumulated audio is emitted regardless of silence.

    Raises
    ------
    RuntimeError
        If openWakeWord model loading fails.

    Notes
    -----
    - SpeexDSP is x86-only; on ARM64 the XMOS DSP handles noise
      suppression in hardware.
    - The ONNX Runtime CUDA warning on Pi is harmless.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        on_wake: Callable[[str, float], None],
        on_utterance: Callable[[np.ndarray], None] | None = None,
        *,
        model_paths: list[str] | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
        input_device: str | int | None = None,
        enable_vad: bool = True,
        cooldown: float = _COOLDOWN_SECONDS,
        capture_silence_duration: float = _CAPTURE_SILENCE_DURATION,
        capture_silence_threshold: float = _CAPTURE_SILENCE_THRESHOLD,
        capture_timeout: float = _CAPTURE_TIMEOUT,
    ) -> None:
        """Initialise the listener.

        See the class docstring for parameter descriptions.
        """
        self._on_wake = on_wake
        self._on_utterance = on_utterance
        self._threshold = threshold
        self._cooldown = cooldown

        # Capture parameters.
        self._capture_silence_duration: float = capture_silence_duration
        self._capture_silence_threshold: float = capture_silence_threshold
        self._capture_timeout: float = capture_timeout

        # Derived capture constants (frame-based for determinism).
        self._silence_window_frames: int = max(
            1, int(capture_silence_duration / (_BLOCK_SIZE / _SAMPLE_RATE))
        )
        self._min_speech_frames: int = max(
            1, int(_CAPTURE_MIN_SPEECH * _SAMPLE_RATE / _BLOCK_SIZE)
        )
        self._max_capture_frames: int = int(
            capture_timeout * _SAMPLE_RATE / _BLOCK_SIZE
        )

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

        # Capture state.  Written from the audio callback thread.
        self._capturing: bool = False
        self._capture_frames: list[np.ndarray] = []
        self._capture_rms: deque[float] = deque(maxlen=self._silence_window_frames)

        # Runtime state.
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
            "Listening for wake words: %s (threshold=%.2f, device=%s, "
            "capture_silence=%.1fs, capture_timeout=%.1fs)",
            self._model_names,
            self._threshold,
            self._device,
            self._capture_silence_duration,
            self._capture_timeout,
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

        Two modes, selected by ``self._capturing``:

        *Listening mode* — feeds audio to openWakeWord and checks
        predictions against threshold, VAD, and cooldown.  On a
        detection, calls ``on_wake`` and enters capture mode.

        *Capture mode* — accumulates audio frames, computes RMS
        energy, and monitors for silence.  When endpoint conditions
        are met (minimum speech recorded + silence window full of
        quiet frames, or timeout reached), calls ``on_utterance``
        and returns to listening mode.

        Parameters
        ----------
        indata:
            Audio samples of shape ``(frames, channels)`` with dtype
            ``int16``.  For mono: ``(1280, 1)``.
        frames:
            Number of sample frames in this block (``_BLOCK_SIZE``).
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

        if self._capturing:
            self._capture_frame(audio)
        else:
            self._detect_wake_word(audio)

    # ------------------------------------------------------------------
    # Listening mode
    # ------------------------------------------------------------------

    def _detect_wake_word(self, audio: np.ndarray) -> None:
        """Run openWakeWord on one frame and check for detections.

        Called in listening mode.  On a valid detection, fires
        ``on_wake`` and enters capture mode.

        Parameters
        ----------
        audio:
            1-D ``int16`` array of ``_BLOCK_SIZE`` samples.
        """
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

            # Enter capture mode if an on_utterance callback is registered.
            entered_capture = False
            if self._on_utterance is not None:
                self._enter_capture_mode()
                entered_capture = True

            # If we entered capture mode, stop checking additional
            # models (the next frame belongs to the utterance).
            if entered_capture:
                break

    # ------------------------------------------------------------------
    # Capture mode
    # ------------------------------------------------------------------

    def _enter_capture_mode(self) -> None:
        """Switch to capture mode, resetting frame/RMS state.

        The current frame (the one that triggered detection) is NOT
        included in the capture buffer — the utterance starts after
        the wake word.
        """
        self._capturing = True
        self._capture_frames.clear()
        self._capture_rms.clear()
        logger.debug("Entered capture mode (max %d frames, silence=%d frames)",
                     self._max_capture_frames, self._silence_window_frames)

    def _capture_frame(self, audio: np.ndarray) -> None:
        """Accumulate one frame and check for endpoint conditions.

        Parameters
        ----------
        audio:
            1-D ``int16`` array of ``_BLOCK_SIZE`` samples.
        """
        self._capture_frames.append(audio.copy())

        # RMS energy for this frame.
        rms: float = float(
            np.sqrt(np.mean(audio.astype(np.float64) ** 2)) / 32768.0
        )
        self._capture_rms.append(rms)

        # Check endpoint conditions.
        ended: bool = False
        reason: str = ""

        # Timeout.
        if len(self._capture_frames) >= self._max_capture_frames:
            ended = True
            reason = "timeout"
        # Silence endpoint.
        elif (
            len(self._capture_frames) > self._min_speech_frames
            and len(self._capture_rms) == self._silence_window_frames
            and all(v < self._capture_silence_threshold for v in self._capture_rms)
        ):
            ended = True
            reason = "silence"

        if ended:
            self._emit_utterance()

    def _emit_utterance(self) -> None:
        """Concatenate captured frames and call ``on_utterance``.

        Returns to listening mode afterwards.  If no frames were
        captured (shouldn't happen), the callback is not invoked.
        """
        self._capturing = False

        if not self._capture_frames:
            logger.debug("No frames captured; returning to listening mode")
            return

        utterance: np.ndarray = np.concatenate(self._capture_frames)
        total_seconds: float = len(utterance) / _SAMPLE_RATE
        logger.info(
            "Utterance captured: %.2fs (%d frames)",
            total_seconds,
            len(self._capture_frames),
        )

        self._capture_frames.clear()
        self._capture_rms.clear()

        if self._on_utterance is not None:
            try:
                self._on_utterance(utterance)
            except Exception:
                logger.exception(
                    "Unhandled exception in on_utterance callback"
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
    If no ``--model`` is provided, all built-in models are loaded
    (including ``hey_jarvis``).

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
