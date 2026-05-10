"""
Speech-to-text transcription using faster-whisper.

Captures audio from the ReSpeaker Lite microphone after a wake word
trigger, detects silence to end the utterance, and transcribes using
a Whisper model via CTranslate2 acceleration. Runs entirely on CPU.

Supports custom word biasing (Whisper prompt injection) and optional
disfluency filtering — confidence-gated (Level 3) with regex fallback
(Level 2).

Architecture:
    Wake word detected
        → record audio from ReSpeaker Lite
        → silence detection (RMS energy threshold)
        → faster-whisper (Whisper → CTranslate2 → ONNX → CPU)
            with custom word prompt biasing and word_timestamps
        → disfluency filter (confidence or regex)
        → return TranscriptionResult (raw_text + clean_text)

Usage:
    from minty_box.stt import SpeechToText

    stt = SpeechToText(
        model_size="tiny.en",
        custom_words=["Araminta", "Minty", "Moorside", "Conford", "Liphook"],
        disfluency_filter="confidence",  # or "regex", "both", None
    )

    result = stt.transcribe_buffer(audio_array)
    print(result.text)       # cleaned
    print(result.raw_text)    # unfiltered
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Optional

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

# Disfluency filter defaults.
_DEFAULT_FILTER_MODE: str = "confidence"  # "confidence" | "regex" | "both"
_DEFAULT_CONFIDENCE_THRESHOLD: float = -1.0  # logprob threshold for confidence mode

# Default filler words for regex mode — matched with word boundaries.
_DEFAULT_FILLER_WORDS: list[str] = [
    "um",
    "uh",
    "er",
    "hmm",
    "mm",
    "ah",
    "erm",
    "I mean",
    "you know",
    "sort of",
    "kind of",
    "I guess",
    "like",
    "actually",
    "basically",
    "literally",
    "right",
    "okay so",
    "so",
    "well",
]

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DisfluencyFilterMode = Literal["confidence", "regex", "both"]


@dataclass
class TranscriptionResult:
    """Result of a speech-to-text transcription.

    Attributes
    ----------
    text:
        Cleaned transcribed text after disfluency filtering.  If no
        filter is active, identical to ``raw_text``.
    raw_text:
        Unfiltered transcribed text as returned by Whisper.
    segments:
        List of individual segment texts (sentences / phrases) after
        filtering.
    raw_segments:
        Unfiltered segment texts.
    language:
        Detected language code (e.g., ``"en"``).
    language_probability:
        Confidence of language detection (0–1).
    duration:
        Duration of the transcribed audio in seconds.
    filter_applied:
        Which disfluency filter was used, if any.
    """

    text: str
    raw_text: str = ""
    segments: list[str] = field(default_factory=list)
    raw_segments: list[str] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    filter_applied: Optional[str] = None


# ---------------------------------------------------------------------------
# SpeechToText
# ---------------------------------------------------------------------------


class SpeechToText:
    """Speech-to-text using faster-whisper on CPU.

    Loads a quantised Whisper model (:mod:`faster_whisper`) and provides
    methods for transcribing pre-recorded audio buffers or recording
    directly from a microphone with silence-based endpoint detection.

    Supports custom word biasing via Whisper's ``prompt`` parameter and
    configurable disfluency filtering.

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
    custom_words:
        Domain-specific words to bias Whisper toward.  Passed as the
        ``prompt`` parameter to :meth:`faster_whisper.WhisperModel.transcribe`.
        Not a hard vocabulary constraint, but strongly biases decoding
        toward these tokens.  Useful for names, places, and technical
        terms that Whisper might otherwise mangle.
    disfluency_filter:
        Which disfluency filtering mode to apply.  ``"confidence"``
        uses token-level logprobs (requires ``word_timestamps=True``,
        slightly slower).  ``"regex"`` scrubs known filler words with
        word-boundary patterns.  ``"both"`` runs confidence first, then
        regex on the result.  ``None`` disables filtering entirely
        (``text`` == ``raw_text``).
    confidence_threshold:
        Logprob threshold for confidence-based filtering.  Tokens with
        log probability below this value are dropped.  Default ``-1.0``
        is a reasonable starting point; lower values are more
        aggressive.  Only meaningful when ``disfluency_filter`` is
        ``"confidence"`` or ``"both"``.
    filler_words:
        Custom list of filler words/phrases for regex mode.  Matched
        case-insensitively with word boundaries.  Defaults to
        `_DEFAULT_FILLER_WORDS` if not provided.

    Notes
    -----
    - The first call to :meth:`transcribe_buffer` or
      :meth:`listen_and_transcribe` triggers model download if not
      already cached.  Models are cached by :mod:`faster_whisper` in
      the HuggingFace Hub cache (~/.cache/huggingface/).
    - ``tiny.en`` is ~75 MB and transcribes ~10x faster than real-time
      on a Pi 5.  ``base.en`` is ~150 MB and ~5x real-time.
    - Confidence mode requires ``word_timestamps=True``, which adds a
      small overhead (~10% slower) for the alignment pass.
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

    _FILTER_MODES: frozenset[str] = frozenset({"confidence", "regex", "both"})

    def __init__(
        self,
        model_size: str = _DEFAULT_MODEL_SIZE,
        device: str = "cpu",
        compute_type: str = _DEFAULT_COMPUTE_TYPE,
        input_device: str | int | None = None,
        silence_duration: float = _DEFAULT_SILENCE_DURATION,
        silence_threshold: float = _DEFAULT_SILENCE_THRESHOLD,
        beam_size: int = 1,
        custom_words: list[str] | None = None,
        disfluency_filter: DisfluencyFilterMode | None = _DEFAULT_FILTER_MODE,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        filler_words: list[str] | None = None,
    ) -> None:
        """Initialise the speech-to-text engine.

        See the class docstring for parameter descriptions.

        Raises
        ------
        ValueError
            If ``model_size`` is not a recognised Whisper model size,
            or if ``disfluency_filter`` is not a valid mode.
        """
        if model_size not in self._AVAILABLE_SIZES:
            raise ValueError(
                f"Unknown model size '{model_size}'. "
                f"Choose from: {', '.join(sorted(self._AVAILABLE_SIZES))}"
            )

        if disfluency_filter is not None and disfluency_filter not in self._FILTER_MODES:
            raise ValueError(
                f"Unknown filter mode '{disfluency_filter}'. "
                f"Choose from: {', '.join(sorted(self._FILTER_MODES))} or None"
            )

        self._model_size: str = model_size
        self._silence_duration: float = silence_duration
        self._silence_threshold: float = silence_threshold
        self._beam_size: int = beam_size

        # Custom word biasing.
        self._custom_words: list[str] = custom_words or []
        self._prompt: str | None = self._build_prompt() if self._custom_words else None
        if self._prompt:
            logger.debug("Prompt: %r", self._prompt)

        # Disfluency filtering.
        self._disfluency_filter: DisfluencyFilterMode | None = disfluency_filter
        self._confidence_threshold: float = confidence_threshold
        self._filler_words: list[str] = filler_words or _DEFAULT_FILLER_WORDS

        # Precompile regex patterns for word-boundary matching.
        # Each filler word becomes \b<word>\b (case-insensitive).
        # Multi-word phrases are handled as literal sequences.
        self._filler_pattern: re.Pattern[str] | None = None
        if self._disfluency_filter in ("regex", "both"):
            self._filler_pattern = self._compile_filler_pattern()

        # Need word timestamps for confidence-based filtering.
        self._need_word_timestamps: bool = (
            self._disfluency_filter in ("confidence", "both")
        )

        # Resolve audio device.
        self._device: str | int = input_device or self._find_respeaker()
        logger.info("STT using device: %s", self._device)

        # Load the Whisper model.
        logger.info(
            "Loading Whisper model: %s (device=%s, compute_type=%s, filter=%s)",
            model_size,
            device,
            compute_type,
            self._disfluency_filter or "none",
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
            Transcribed text with both raw and cleaned versions,
            segments, language, and duration.

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

        # Build transcribe kwargs.
        transcribe_kwargs: dict = {
            "beam_size": self._beam_size,
            "language": "en",
            "vad_filter": False,
        }
        if self._prompt:
            transcribe_kwargs["initial_prompt"] = self._prompt
        if self._need_word_timestamps:
            transcribe_kwargs["word_timestamps"] = True

        segments_raw, info = self._model.transcribe(audio_f32, **transcribe_kwargs)

        # Materialise raw segments.
        raw_segments: list[str] = []
        for seg in segments_raw:
            raw_segments.append(seg.text.strip())

        elapsed: float = time.monotonic() - start
        raw_text: str = " ".join(raw_segments).strip()

        # Apply disfluency filtering.
        clean_text: str
        clean_segments: list[str]
        filter_applied: Optional[str]

        if self._disfluency_filter and raw_text:
            clean_text, clean_segments, filter_applied = self._filter(
                raw_text, raw_segments, segments_raw  # type: ignore[arg-type]
            )
        else:
            clean_text, clean_segments = raw_text, raw_segments
            filter_applied = None

        logger.info(
            "Transcribed %.1fs audio in %.2fs (%d segments, lang=%s, filter=%s)",
            info.duration,
            elapsed,
            len(clean_segments),
            info.language,
            filter_applied or "none",
        )

        return TranscriptionResult(
            text=clean_text,
            raw_text=raw_text,
            segments=clean_segments,
            raw_segments=raw_segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            filter_applied=filter_applied,
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
    # Custom word biasing
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str | None:
        """Build a Whisper prompt string from the custom word list.

        The prompt is passed as previous-text context, which biases the
        decoder toward these tokens without breaking the language model.
        Format: comma-separated list in a natural-looking sentence.

        Returns
        -------
        str or None
            Prompt string, or ``None`` if the word list is empty.
        """
        if not self._custom_words:
            return None
        return " ".join(self._custom_words)

    # ------------------------------------------------------------------
    # Disfluency filtering
    # ------------------------------------------------------------------

    def _filter(
        self,
        raw_text: str,
        raw_segments: list[str],
        segments_raw: list,  # list of faster_whisper.transcribe.Segment
    ) -> tuple[str, list[str], str]:
        """Apply configured disfluency filter.

        Parameters
        ----------
        raw_text:
            Full unfiltered text.
        raw_segments:
            Per-segment unfiltered texts.
        segments_raw:
            Original faster-whisper Segment objects (needed for token
            probabilities in confidence mode).

        Returns
        -------
        (clean_text, clean_segments, filter_applied)
            Cleaned text, cleaned segment list, and a label describing
            which filter(s) were applied.
        """
        mode = self._disfluency_filter

        if mode == "confidence":
            return (*self._filter_confidence(raw_segments, segments_raw), "confidence")
        elif mode == "regex":
            return (*self._filter_regex(raw_segments), "regex")
        elif mode == "both":
            # Confidence first, then regex on the result.
            texts_conf, segs_conf = self._filter_confidence(raw_segments, segments_raw)
            texts_regex, segs_regex = self._filter_regex(segs_conf)
            return texts_regex, segs_regex, "confidence+regex"
        else:
            return raw_text, raw_segments, "none"

    def _filter_confidence(
        self,
        raw_segments: list[str],
        segments_raw: list,
    ) -> tuple[str, list[str]]:
        """Filter disfluencies using token-level log probabilities.

        Iterates each segment's word list and drops tokens with logprob
        below ``_confidence_threshold``.  Surviving words are rejoined
        to form cleaned segments.

        Falls back to returning ``raw_segments`` unchanged if the
        segment objects lack word-level probability data (e.g., if
        ``word_timestamps`` was not enabled).

        Parameters
        ----------
        raw_segments:
            Per-segment unfiltered texts (fallback if no word data).
        segments_raw:
            Original faster-whisper Segment objects with ``words``
            attribute containing ``(word, start, end, probability)``
            tuples.

        Returns
        -------
        (clean_text, clean_segments)
            Filtered text and segments.
        """
        clean_segments: list[str] = []

        for seg in segments_raw:
            words: list = getattr(seg, "words", None) or []

            if not words:
                # No word-level data — keep segment unchanged.
                clean_segments.append(seg.text.strip())
                continue

            # Keep words above the logprob threshold.
            kept: list[str] = []
            for word_entry in words:
                # word_entry is (word, start, end, probability) or a
                # named tuple / object with .word and .probability
                if hasattr(word_entry, "word"):
                    word_text: str = word_entry.word
                    word_prob: float = word_entry.probability
                else:
                    word_text = word_entry[0]
                    word_prob = word_entry[3] if len(word_entry) > 3 else 0.0

                if word_prob >= self._confidence_threshold:
                    kept.append(word_text)

            if kept:
                # Join with spaces, then fix common spacing issues.
                joined = " ".join(kept).strip()
                # Collapse whitespace around punctuation.
                joined = re.sub(r"\s+([.,!?;:])", r"\1", joined)
                clean_segments.append(joined)
            else:
                # All words dropped — keep original segment as fallback.
                clean_segments.append(seg.text.strip())

        clean_text: str = " ".join(clean_segments).strip()
        return clean_text, clean_segments

    def _filter_regex(
        self, segments: list[str]
    ) -> tuple[str, list[str]]:
        """Filter disfluencies using word-boundary regex.

        Removes known filler words/phrases from each segment.  Also
        cleans up punctuation artifacts left behind by word removal
        (leading commas, double punctuation, etc.).

        Parameters
        ----------
        segments:
            Segment texts to clean.

        Returns
        -------
        (clean_text, clean_segments)
            Filtered text and segments.
        """
        if self._filler_pattern is None:
            clean_segments: list[str] = list(segments)
        else:
            clean_segments = []
            for seg in segments:
                cleaned: str = self._filler_pattern.sub("", seg)
                # Clean up punctuation artifacts from removed words.
                # Leading/trailing commas after filler removal.
                cleaned = re.sub(r"^\s*,\s*", "", cleaned)
                cleaned = re.sub(r"\s*,\s*$", "", cleaned)
                # Collapse double commas.
                cleaned = re.sub(r",\s*,", ",", cleaned)
                # Collapse multiple spaces.
                cleaned = re.sub(r"\s{2,}", " ", cleaned)
                # Fix ", and" or ", but" after filler removal.
                cleaned = re.sub(r",\s+(and|but|or|so)\b", r" \1", cleaned)
                cleaned = cleaned.strip()
                clean_segments.append(cleaned)

        clean_text: str = " ".join(clean_segments).strip()
        return clean_text, clean_segments

    def _compile_filler_pattern(self) -> re.Pattern[str]:
        """Compile a regex from the filler word list.

        Each filler word is wrapped in ``\\b...\\b`` for word-boundary
        matching.  Multi-word phrases are handled correctly (the phrase
        itself forms a unit).  The pattern is case-insensitive.

        Returns
        -------
        re.Pattern
            Compiled regex.
        """
        # Sort by length descending so longer phrases match first
        # (e.g., "I mean" before "I" if "I" were in the list).
        sorted_words: list[str] = sorted(
            self._filler_words, key=len, reverse=True
        )

        patterns: list[str] = []
        for word in sorted_words:
            # Escape regex special chars, then wrap.
            escaped: str = re.escape(word)
            patterns.append(r"\b" + escaped + r"\b")

        combined: str = "|".join(patterns)
        return re.compile(combined, re.IGNORECASE)

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
