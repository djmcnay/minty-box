"""
Speech-to-text transcription using faster-whisper.

Captures audio from the ReSpeaker Lite microphone after a wake word
trigger, detects silence to end the utterance, and transcribes using
a Whisper model via CTranslate2 acceleration. Runs entirely on CPU.

Architecture:
    Wake word detected
        → record audio from ReSpeaker Lite
        → silence detection (RMS energy threshold)
        → faster-whisper (Whisper → CTranslate2 → ONNX → CPU)
        → return transcribed text

Usage:
    from minty_box.stt import SpeechToText

    stt = SpeechToText(model_size="tiny.en")

    # Transcribe a pre-recorded buffer
    text = stt.transcribe_buffer(audio_array)

    # Record from mic until silence, then transcribe
    text = stt.listen_and_transcribe(timeout=10.0)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Substring used to identify the ReSpeaker Lite for recording.
_RESPEAKER_NAME: str = "ReSpeaker Lite"

# Audio format.
_SAMPLE_RATE: int = 16000
_CHANNELS: int = 1
_BLOCK_SIZE: int = 1280  # 80 ms frames
_DTYPE: str = "int16"

# Silence detection defaults.
_DEFAULT_SILENCE_DURATION: float = 1.5  # seconds of quiet before endpointing
_DEFAULT_SILENCE_THRESHOLD: float = 0.01  # RMS amplitude threshold
_DEFAULT_TIMEOUT: float = 15.0  # maximum recording duration

# Model defaults.
_DEFAULT_MODEL_SIZE: str = "tiny.en"
_DEFAULT_COMPUTE_TYPE: str = "int8"

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """Result of a speech-to-text transcription.

    Attributes
    ----------
    text:
        Full transcribed text with segments joined.
    segments:
        List of individual segment texts (sentences / phrases).
    language:
        Detected language code (e.g., ``"en"``).
    language_probability:
        Confidence of language detection (0–1).
    duration:
        Duration of the transcribed audio in seconds.
    """

    text: str
    segments: list[str] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0


# ---------------------------------------------------------------------------
# SpeechToText
# ---------------------------------------------------------------------------


class SpeechToText:
    """Speech-to-text using faster-whisper on CPU.

    Loads a quantised Whisper model (:mod:`faster_whisper`) and provides
    methods for transcribing pre-recorded audio buffers or recording
    directly from a microphone with silence-based endpoint detection.

    Parameters
    ----------
    model_size:
        Whisper model size.  One of ``"tiny"``, ``"tiny.en"``,
        ``"base"``, ``"base.en"``, ``"small"``, ``"small.en"``,
        ``"medium"``, ``"medium.en"``, ``"large-v3"``.
        English-only models (``.en``) are faster and sufficient for
        single-language use.
    device:
        Compute device.  Always ``"cpu"`` on Raspberry Pi.
    compute_type:
        Quantisation type for CTranslate2.  ``"int8"`` is the best
        balance of speed and accuracy on ARM64 CPU.
    input_device:
        sounddevice device identifier for recording.  If ``None``,
        auto-detects ReSpeaker Lite.
    silence_duration:
        Seconds of continuous silence before ending the utterance
        during :meth:`listen_and_transcribe`.
    silence_threshold:
        RMS amplitude threshold below which audio is considered
        silence.  Calibrate for your environment — quiet cottages
        may want a lower value, noisy rooms a higher one.
    beam_size:
        Beam search width during transcription.  Larger values improve
        accuracy at the cost of speed.  Default 1 is greedy decoding
        (fastest).

    Notes
    -----
    - The first call to :meth:`transcribe_buffer` or
      :meth:`listen_and_transcribe` triggers model download if not
      already cached.  Models are cached by :mod:`faster_whisper` in
      the HuggingFace Hub cache (~/.cache/huggingface/).
    - ``tiny.en`` is ~75 MB and transcribes ~10x faster than real-time
      on a Pi 5.  ``base.en`` is ~150 MB and ~5x real-time.
    """

    _AVAILABLE_SIZES: frozenset[str] = frozenset(
        {
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large-v3",
        }
    )

    def __init__(
        self,
        model_size: str = _DEFAULT_MODEL_SIZE,
        device: str = "cpu",
        compute_type: str = _DEFAULT_COMPUTE_TYPE,
        input_device: str | int | None = None,
        silence_duration: float = _DEFAULT_SILENCE_DURATION,
        silence_threshold: float = _DEFAULT_SILENCE_THRESHOLD,
        beam_size: int = 1,
    ) -> None:
        """Initialise the speech-to-text engine.

        See the class docstring for parameter descriptions.

        Raises
        ------
        ValueError
            If ``model_size`` is not a recognised Whisper model size.
        """
        if model_size not in self._AVAILABLE_SIZES:
            raise ValueError(
                f"Unknown model size '{model_size}'. "
                f"Choose from: {', '.join(sorted(self._AVAILABLE_SIZES))}"
            )

        self._model_size: str = model_size
        self._silence_duration: float = silence_duration
        self._silence_threshold: float = silence_threshold
        self._beam_size: int = beam_size

        # Resolve audio device.
        self._device: str | int = input_device or self._find_respeaker()
        logger.info("STT using device: %s", self._device)

        # Load the Whisper model (lazy — actual download happens on
        # first inference, not here, but this validates parameters).
        logger.info(
            "Loading Whisper model: %s (device=%s, compute_type=%s)",
            model_size,
            device,
            compute_type,
        )
        self._model: WhisperModel = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_buffer(
        self,
        audio: np.ndarray,
        sample_rate: int = _SAMPLE_RATE,
    ) -> TranscriptionResult:
        """Transcribe a pre-recorded audio buffer.

        Parameters
        ----------
        audio:
            1-D NumPy array of 16-bit PCM samples.
            Will be converted to float32 internally.
        sample_rate:
            Sample rate of ``audio`` in Hz.  Must be 16000.

        Returns
        -------
        TranscriptionResult
            Transcribed text, segments, language, and duration.

        Raises
        ------
        ValueError
            If ``audio`` is empty.
        """
        if audio.size == 0:
            raise ValueError("Audio buffer is empty")

        # Convert int16 → float32, normalised to [-1, 1].
        if audio.dtype == np.int16:
            audio_f32: np.ndarray = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32:
            audio_f32 = audio
        else:
            audio_f32 = audio.astype(np.float32)

        start: float = time.monotonic()

        segments_raw, info = self._model.transcribe(
            audio_f32,
            beam_size=self._beam_size,
            language="en",  # English-only for speed
            vad_filter=False,  # We handle VAD/silence ourselves upstream
        )

        # Materialise segments (generator → list).
        segments: list[str] = []
        for seg in segments_raw:
            segments.append(seg.text.strip())

        elapsed: float = time.monotonic() - start

        full_text: str = " ".join(segments).strip()

        logger.info(
            "Transcribed %.1fs audio in %.2fs (%s segments, lang=%s)",
            info.duration,
            elapsed,
            len(segments),
            info.language,
        )

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )

    def listen_and_transcribe(
        self,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> TranscriptionResult:
        """Record audio from the microphone until silence, then transcribe.

        Opens an input stream, accumulates audio frames, and monitors
        the RMS energy.  When the energy stays below
        ``silence_threshold`` for ``silence_duration`` seconds
        continuously, recording stops and the buffer is transcribed.
        A ``timeout`` prevents indefinite recording if the user walks
        away.

        Parameters
        ----------
        timeout:
            Maximum recording duration in seconds.  If reached, the
            accumulated audio is transcribed regardless of silence.

        Returns
        -------
        TranscriptionResult
            Transcribed text from the recorded utterance.

        Notes
        -----
        This method blocks until recording completes.  It is designed
        to be called directly from a wake word callback.
        """
        logger.info(
            "Listening for speech... (timeout=%.1fs, "
            "silence_duration=%.1fs, threshold=%.4f)",
            timeout,
            self._silence_duration,
            self._silence_threshold,
        )

        # Rolling window for silence detection.
        # Each frame is 80 ms; window covers silence_duration seconds.
        window_frames: int = max(
            1, int(self._silence_duration / (_BLOCK_SIZE / _SAMPLE_RATE))
        )
        recent_rms: deque[float] = deque(maxlen=window_frames)

        # Minimum speech: require at least 0.5s of audio before
        # allowing silence to end the recording.  Measured in frames
        # rather than wall-clock time so tests run deterministically.
        min_speech_frames: int = max(
            1, int(0.5 * _SAMPLE_RATE / _BLOCK_SIZE)
        )

        # Accumulated audio.
        frames: list[np.ndarray] = []

        frames_per_second: float = _SAMPLE_RATE / _BLOCK_SIZE
        max_frames: int = int(timeout * frames_per_second)

        start_time: float = time.monotonic()

        with sd.InputStream(
            device=self._device,
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype=_DTYPE,
            blocksize=_BLOCK_SIZE,
        ) as stream:
            while len(frames) < max_frames:
                data, _overflowed = stream.read(_BLOCK_SIZE)
                audio_block: np.ndarray = data[:, 0]
                frames.append(audio_block)

                # RMS energy for this block.
                rms: float = float(np.sqrt(np.mean(audio_block.astype(np.float64) ** 2)) / 32768.0)
                recent_rms.append(rms)

                # Silence endpoint: window is full, minimum speech
                # frames have been recorded, and all values in the
                # window are below threshold.
                if (
                    len(frames) > min_speech_frames
                    and len(recent_rms) == window_frames
                    and all(v < self._silence_threshold for v in recent_rms)
                ):
                    logger.debug(
                        "Silence detected after %d frames (%d quiet frames)",
                        len(frames),
                        window_frames,
                    )
                    break

        elapsed_total: float = time.monotonic() - start_time
        logger.info(
            "Recorded %.2fs of audio (%d frames)",
            elapsed_total,
            len(frames),
        )

        if not frames:
            logger.warning("No audio recorded")
            return TranscriptionResult(text="")

        audio: np.ndarray = np.concatenate(frames)
        return self.transcribe_buffer(audio)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_respeaker() -> str | int:
        """Locate the ReSpeaker Lite in sounddevice's device list.

        Returns
        -------
        str | int
            sounddevice device identifier.  Integer index if found,
            system default input device otherwise.
        """
        devices: list[dict] = sd.query_devices()
        for idx, dev in enumerate(devices):
            if _RESPEAKER_NAME in dev["name"]:
                return idx
        logger.warning(
            "ReSpeaker Lite not found. Using default. Devices: %s",
            [d["name"] for d in devices],
        )
        return sd.default.device[0]  # type: ignore[return-value]
