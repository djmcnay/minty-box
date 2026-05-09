"""Tests for minty_box.stt."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from minty_box import SpeechToText, TranscriptionResult
from minty_box.stt import (
    _BLOCK_SIZE,
    _CHANNELS,
    _DEFAULT_CONFIDENCE_THRESHOLD,
    _DEFAULT_FILLER_WORDS,
    _DEFAULT_FILTER_MODE,
    _DEFAULT_SILENCE_DURATION,
    _DEFAULT_SILENCE_THRESHOLD,
    _DEFAULT_TIMEOUT,
    _DTYPE,
    _RESPEAKER_NAME,
    _SAMPLE_RATE,
    _DEFAULT_MODEL_SIZE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_whisper_model() -> MagicMock:
    """Return a mock WhisperModel that returns canned transcription output."""
    model = MagicMock()

    segment = MagicMock()
    segment.text = "hello world"
    segment.words = None  # no word-level data by default

    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.99
    info.duration = 2.0

    model.transcribe.return_value = ([segment], info)
    return model


def _make_word(word: str, prob: float) -> MagicMock:
    """Build a mock word entry with ``word`` and ``probability`` attrs."""
    w = MagicMock()
    w.word = word
    w.probability = prob
    return w


@pytest.fixture
def speech_audio() -> np.ndarray:
    """Return 1 second of loud synthetic audio (sine wave)."""
    t = np.linspace(0, 1.0, _SAMPLE_RATE, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for :class:`SpeechToText` initialisation."""

    def test_valid_model_sizes_accepted(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """All recognised model sizes construct without error."""
        valid = ["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                 "medium", "medium.en", "large-v3"]
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            for size in valid:
                stt = SpeechToText(model_size=size)
                assert stt._model_size == size

    def test_invalid_model_size_raises(self) -> None:
        """Unknown model sizes raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model size"):
            SpeechToText(model_size="gigantic")

    def test_invalid_filter_mode_raises(self) -> None:
        """Unknown filter modes raise ValueError."""
        with pytest.raises(ValueError, match="Unknown filter mode"):
            SpeechToText(disfluency_filter="magic")  # type: ignore[arg-type]

    def test_custom_silence_params_stored(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Threshold and duration are stored."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(silence_threshold=0.05, silence_duration=2.0)
            assert stt._silence_threshold == 0.05
            assert stt._silence_duration == 2.0

    def test_default_params(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Default params match module constants."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText()
            assert stt._silence_threshold == _DEFAULT_SILENCE_THRESHOLD
            assert stt._silence_duration == _DEFAULT_SILENCE_DURATION
            assert stt._disfluency_filter == _DEFAULT_FILTER_MODE
            assert stt._confidence_threshold == _DEFAULT_CONFIDENCE_THRESHOLD

    def test_custom_device(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Explicit device string is respected."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(input_device="pulse")
            assert stt._device == "pulse"

    # ------------------------------------------------------------------
    # Custom words
    # ------------------------------------------------------------------

    def test_custom_words_build_prompt(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Custom words produce a prompt string for Whisper biasing."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(
                custom_words=["Araminta", "Minty", "Moorside"],
            )
            assert stt._prompt == "Araminta Minty Moorside"

    def test_empty_custom_words_no_prompt(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """No prompt when custom_words is empty or None."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(custom_words=[])
            assert stt._prompt is None

            stt2 = SpeechToText()
            assert stt2._prompt is None

    def test_filter_none_disables_everything(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """disfluency_filter=None → no prompt needed, no word_timestamps."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(disfluency_filter=None)
            assert stt._need_word_timestamps is False
            assert stt._filler_pattern is None


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for :meth:`SpeechToText._build_prompt`."""

    def test_single_word(self, mock_whisper_model: MagicMock) -> None:
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(custom_words=["Araminta"])
            assert stt._build_prompt() == "Araminta"

    def test_empty_returns_none(self, mock_whisper_model: MagicMock) -> None:
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(custom_words=[])
            assert stt._build_prompt() is None


# ---------------------------------------------------------------------------
# _compile_filler_pattern
# ---------------------------------------------------------------------------


class TestCompileFillerPattern:
    """Tests for :meth:`SpeechToText._compile_filler_pattern`."""

    def test_pattern_matches_filler_words(self, mock_whisper_model: MagicMock) -> None:
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(disfluency_filter="regex")
            p = stt._compile_filler_pattern()

            # Matches standalone filler words.
            assert p.search("um, what time is it")
            # Does not match substring inside larger words.
            assert not p.search("umbrella")
            assert not p.search("human")
            # Case insensitive.
            assert p.search("Um, okay")
            # Multi-word phrase.
            assert p.search("I mean, that's fine")

    def test_longer_phrases_match_first(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Longer filler phrases like 'I mean' are matched before shorter ones."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(
                disfluency_filter="regex",
                filler_words=["I", "I mean"],
            )
            p = stt._compile_filler_pattern()
            # "I mean" should be matched as a whole phrase.
            match = p.search("I mean, okay")
            assert match is not None
            assert match.group().strip() == "I mean"


# ---------------------------------------------------------------------------
# transcribe_buffer — core behaviour
# ---------------------------------------------------------------------------


class TestTranscribeBuffer:
    """Tests for :meth:`SpeechToText.transcribe_buffer`."""

    def test_transcribes_speech(
        self,
        mock_whisper_model: MagicMock,
        speech_audio: np.ndarray,
    ) -> None:
        """Returns TranscriptionResult with the expected text."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(disfluency_filter=None)
            result = stt.transcribe_buffer(speech_audio)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert result.raw_text == "hello world"
        assert result.language == "en"
        assert result.language_probability == 0.99
        assert len(result.segments) == 1
        assert result.filter_applied is None

    def test_empty_buffer_raises(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Empty audio raises ValueError."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(disfluency_filter=None)
            with pytest.raises(ValueError, match="empty"):
                stt.transcribe_buffer(np.array([], dtype=np.int16))

    def test_converts_int16_to_float32(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Audio is normalised to float32 [-1, 1] before model call."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(disfluency_filter=None)
            audio = np.array([16384, -16384, 0], dtype=np.int16)
            stt.transcribe_buffer(audio)

        called_audio = mock_whisper_model.transcribe.call_args[0][0]
        assert called_audio.dtype == np.float32
        assert np.all(called_audio >= -1.0)
        assert np.all(called_audio <= 1.0)

    def test_multi_segment_transcription(
        self, mock_whisper_model: MagicMock, speech_audio: np.ndarray,
    ) -> None:
        """Multiple segments are joined correctly."""
        seg1 = MagicMock()
        seg1.text = "hello"
        seg1.words = None
        seg2 = MagicMock()
        seg2.text = "world"
        seg2.words = None
        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        info.duration = 3.0
        mock_whisper_model.transcribe.return_value = ([seg1, seg2], info)

        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(disfluency_filter=None)
            result = stt.transcribe_buffer(speech_audio)

        assert result.text == "hello world"
        assert result.raw_text == "hello world"
        assert result.segments == ["hello", "world"]

    def test_prompt_passed_to_transcribe(
        self, mock_whisper_model: MagicMock, speech_audio: np.ndarray,
    ) -> None:
        """Custom word prompt is forwarded to model.transcribe()."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(
                custom_words=["Araminta", "Minty"],
                disfluency_filter=None,
            )
            stt.transcribe_buffer(speech_audio)

        kwargs = mock_whisper_model.transcribe.call_args[1]
        assert kwargs.get("prompt") == "Araminta Minty"


# ---------------------------------------------------------------------------
# transcribe_buffer — disfluency filtering
# ---------------------------------------------------------------------------


class TestTranscribeBufferDisfluency:
    """Tests for disfluency filtering during transcription."""

    # ------------------------------------------------------------------
    # Confidence mode
    # ------------------------------------------------------------------

    def test_confidence_drops_low_prob_words(
        self, mock_whisper_model: MagicMock, speech_audio: np.ndarray,
    ) -> None:
        """Words below logprob threshold are removed."""
        seg = MagicMock()
        seg.text = "um hello er world"
        seg.words = [
            _make_word("um", -2.5),
            _make_word("hello", -0.3),
            _make_word("er", -3.0),
            _make_word("world", -0.2),
        ]
        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        info.duration = 2.0
        mock_whisper_model.transcribe.return_value = ([seg], info)

        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(disfluency_filter="confidence", confidence_threshold=-1.0)
            result = stt.transcribe_buffer(speech_audio)

        assert result.text == "hello world"
        assert result.raw_text == "um hello er world"
        assert result.filter_applied == "confidence"

    def test_confidence_keeps_all_when_above_threshold(
        self, mock_whisper_model: MagicMock, speech_audio: np.ndarray,
    ) -> None:
        """All words kept when logprobs are above threshold."""
        seg = MagicMock()
        seg.text = "hello world"
        seg.words = [
            _make_word("hello", -0.3),
            _make_word("world", -0.2),
        ]
        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        info.duration = 2.0
        mock_whisper_model.transcribe.return_value = ([seg], info)

        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(disfluency_filter="confidence", confidence_threshold=-1.0)
            result = stt.transcribe_buffer(speech_audio)

        assert result.text == "hello world"

    def test_confidence_fallback_when_no_word_data(
        self, mock_whisper_model: MagicMock, speech_audio: np.ndarray,
    ) -> None:
        """When segment has no words list, segment text is kept as-is."""
        seg = MagicMock()
        seg.text = "um hello world"
        seg.words = []
        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        info.duration = 2.0
        mock_whisper_model.transcribe.return_value = ([seg], info)

        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(disfluency_filter="confidence")
            result = stt.transcribe_buffer(speech_audio)

        # Falls back to raw segment since no word data.
        assert result.text == "um hello world"

    def test_confidence_all_words_dropped_falls_back(
        self, mock_whisper_model: MagicMock, speech_audio: np.ndarray,
    ) -> None:
        """If all words dropped, keep the original segment."""
        seg = MagicMock()
        seg.text = "um er uh"
        seg.words = [
            _make_word("um", -5.0),
            _make_word("er", -5.0),
            _make_word("uh", -5.0),
        ]
        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        info.duration = 1.0
        mock_whisper_model.transcribe.return_value = ([seg], info)

        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(disfluency_filter="confidence")
            result = stt.transcribe_buffer(speech_audio)

        assert result.text == "um er uh"  # fallback

    # ------------------------------------------------------------------
    # Regex mode
    # ------------------------------------------------------------------

    def test_regex_strips_filler_words(
        self, mock_whisper_model: MagicMock, speech_audio: np.ndarray,
    ) -> None:
        """Known filler words are removed from the text."""
        seg = MagicMock()
        seg.text = "um, hello there, er, world"
        seg.words = None
        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        info.duration = 2.0
        mock_whisper_model.transcribe.return_value = ([seg], info)

        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(disfluency_filter="regex")
            result = stt.transcribe_buffer(speech_audio)

        assert result.text == "hello there, world"
        assert result.filter_applied == "regex"

    def test_regex_preserves_non_filler_words(
        self, mock_whisper_model: MagicMock, speech_audio: np.ndarray,
    ) -> None:
        """Only exact filler words are removed — content words survive."""
        seg = MagicMock()
        seg.text = "set a timer for five minutes"
        seg.words = None
        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        info.duration = 1.0
        mock_whisper_model.transcribe.return_value = ([seg], info)

        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(disfluency_filter="regex")
            result = stt.transcribe_buffer(speech_audio)

        assert result.text == "set a timer for five minutes"

    def test_regex_custom_filler_words(
        self, mock_whisper_model: MagicMock, speech_audio: np.ndarray,
    ) -> None:
        """User-provided filler word list is used instead of defaults."""
        seg = MagicMock()
        seg.text = "blah blah the real content"
        seg.words = None
        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        info.duration = 1.0
        mock_whisper_model.transcribe.return_value = ([seg], info)

        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(
                disfluency_filter="regex",
                filler_words=["blah"],
            )
            result = stt.transcribe_buffer(speech_audio)

        assert result.text == "the real content"

    # ------------------------------------------------------------------
    # Both mode
    # ------------------------------------------------------------------

    def test_both_mode_runs_confidence_then_regex(
        self, mock_whisper_model: MagicMock, speech_audio: np.ndarray,
    ) -> None:
        """Confidence drops low-prob words, then regex scrubs remaining fillers."""
        seg = MagicMock()
        seg.text = "um well hello there er hmm"
        seg.words = [
            _make_word("um", -2.5),
            _make_word("well", -2.0),   # low prob → dropped
            _make_word("hello", -0.3),
            _make_word("there", -0.2),
            _make_word("er", -0.4),     # kept by confidence, but stripped by regex
            _make_word("hmm", -3.0),    # low prob → dropped
        ]
        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        info.duration = 2.0
        mock_whisper_model.transcribe.return_value = ([seg], info)

        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText(
                disfluency_filter="both",
                confidence_threshold=-1.0,
            )
            result = stt.transcribe_buffer(speech_audio)

        assert result.filter_applied == "confidence+regex"
        assert result.text == "hello there"
        assert result.raw_text == "um well hello there er hmm"


# ---------------------------------------------------------------------------
# listen_and_transcribe
# ---------------------------------------------------------------------------


class TestListenAndTranscribe:
    """Tests for :meth:`SpeechToText.listen_and_transcribe`."""

    @staticmethod
    def _make_stt(
        mock_model: MagicMock,
        silence_threshold: float = _DEFAULT_SILENCE_THRESHOLD,
        silence_duration: float = _DEFAULT_SILENCE_DURATION,
    ) -> SpeechToText:
        """Construct a SpeechToText with mocked deps."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            return SpeechToText(
                silence_threshold=silence_threshold,
                silence_duration=silence_duration,
                input_device=0,
                disfluency_filter=None,
            )

    def test_stops_on_silence(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Recording ends when silence is detected."""
        stt = self._make_stt(mock_whisper_model, silence_duration=0.16)

        call_count = [0]

        def fake_stream(**kwargs: Any) -> Any:
            class FakeStream:
                def __enter__(self: Any) -> Any:
                    return self
                def __exit__(self: Any, *args: Any) -> None:
                    pass
                def read(self: Any, frames: int) -> tuple[np.ndarray, bool]:
                    call_count[0] += 1
                    if call_count[0] <= 10:
                        return np.full((_BLOCK_SIZE, 1), 16384, dtype=np.int16), False
                    return np.zeros((_BLOCK_SIZE, 1), dtype=np.int16), False
            return FakeStream()

        with patch("minty_box.stt.sd.InputStream", side_effect=fake_stream):
            result = stt.listen_and_transcribe(timeout=2.0)

        max_frames = int(2.0 * _SAMPLE_RATE / _BLOCK_SIZE)
        assert call_count[0] < max_frames
        assert isinstance(result, TranscriptionResult)

    def test_requires_minimum_speech_before_silence(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Continuous silence at start doesn't immediately exit."""
        stt = self._make_stt(mock_whisper_model, silence_duration=0.08)

        call_count = [0]

        def fake_stream(**kwargs: Any) -> Any:
            class FakeStream:
                def __enter__(self: Any) -> Any:
                    return self
                def __exit__(self: Any, *args: Any) -> None:
                    pass
                def read(self: Any, frames: int) -> tuple[np.ndarray, bool]:
                    call_count[0] += 1
                    if call_count[0] <= 7:
                        return np.zeros((_BLOCK_SIZE, 1), dtype=np.int16), False
                    return np.full((_BLOCK_SIZE, 1), 16384, dtype=np.int16), False
            return FakeStream()

        with patch("minty_box.stt.sd.InputStream", side_effect=fake_stream):
            result = stt.listen_and_transcribe(timeout=5.0)

        assert call_count[0] >= 7
        assert isinstance(result, TranscriptionResult)

    def test_timeout_ends_recording(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Recording ends at timeout even without silence."""
        stt = self._make_stt(mock_whisper_model)

        def fake_stream(**kwargs: Any) -> Any:
            class FakeStream:
                def __enter__(self: Any) -> Any:
                    return self
                def __exit__(self: Any, *args: Any) -> None:
                    pass
                def read(self: Any, frames: int) -> tuple[np.ndarray, bool]:
                    return np.full((_BLOCK_SIZE, 1), 16384, dtype=np.int16), False
            return FakeStream()

        start = time.monotonic()
        with patch("minty_box.stt.sd.InputStream", side_effect=fake_stream):
            result = stt.listen_and_transcribe(timeout=1.0)
        elapsed = time.monotonic() - start

        assert elapsed < 1.5
        assert isinstance(result, TranscriptionResult)

    def test_no_audio_returns_empty(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """If no frames recorded, result.text is ''."""
        stt = self._make_stt(mock_whisper_model)

        original = stt.transcribe_buffer
        stt.transcribe_buffer = MagicMock(
            side_effect=AssertionError("should not be called")
        )  # type: ignore[method-assign]

        def fake_stream(**kwargs: Any) -> Any:
            class FakeStream:
                def __enter__(self: Any) -> Any:
                    return self
                def __exit__(self: Any, *args: Any) -> None:
                    pass
                def read(self: Any, frames: int) -> tuple[np.ndarray, bool]:
                    raise KeyboardInterrupt
            return FakeStream()

        with patch("minty_box.stt.sd.InputStream", side_effect=fake_stream):
            result = stt.listen_and_transcribe(timeout=0.01)

        assert result.text == ""
        stt.transcribe_buffer = original  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# _find_respeaker
# ---------------------------------------------------------------------------


class TestFindRespeakerSTT:
    """Tests for :meth:`SpeechToText._find_respeaker`."""

    def test_finds_respeaker_by_name(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Returns the index of the ReSpeaker Lite device."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd") as mock_sd,
        ):
            mock_sd.query_devices.return_value = [
                {"name": "pulse"},
                {"name": "ReSpeaker Lite: USB Audio (hw:2,0)"},
            ]
            stt = SpeechToText(disfluency_filter=None)
            assert stt._find_respeaker() == 1


# ---------------------------------------------------------------------------
# TranscriptionResult dataclass
# ---------------------------------------------------------------------------


class TestTranscriptionResult:
    """Tests for :class:`TranscriptionResult`."""

    def test_default_values(self) -> None:
        """Fields have correct defaults."""
        result = TranscriptionResult(text="test")
        assert result.text == "test"
        assert result.raw_text == ""
        assert result.segments == []
        assert result.raw_segments == []
        assert result.language == ""
        assert result.language_probability == 0.0
        assert result.duration == 0.0
        assert result.filter_applied is None

    def test_full_construction(self) -> None:
        """All fields can be set explicitly."""
        result = TranscriptionResult(
            text="hello world",
            raw_text="um hello er world",
            segments=["hello world"],
            raw_segments=["um hello er world"],
            language="en",
            language_probability=0.99,
            duration=2.5,
            filter_applied="confidence",
        )
        assert result.text == "hello world"
        assert result.raw_text == "um hello er world"
        assert result.segments == ["hello world"]
        assert result.raw_segments == ["um hello er world"]
        assert result.language == "en"
        assert result.language_probability == 0.99
        assert result.duration == 2.5
        assert result.filter_applied == "confidence"
