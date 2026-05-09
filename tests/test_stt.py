"""Tests for minty_box.stt."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from minty_box import SpeechToText, TranscriptionResult
from minty_box.stt import _DEFAULT_MODEL_SIZE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_whisper_model() -> MagicMock:
    """Return a mock WhisperModel that returns canned transcription output."""
    model = MagicMock()

    # Create a fake segment object.
    segment = MagicMock()
    segment.text = "hello world"

    # Create a fake info object.
    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.99
    info.duration = 2.0

    model.transcribe.return_value = ([segment], info)
    return model


@pytest.fixture
def silent_audio() -> np.ndarray:
    """Return 1 second of silent audio."""
    return np.zeros(_SAMPLE_RATE, dtype=np.int16)


@pytest.fixture
def speech_audio() -> np.ndarray:
    """Return 1 second of loud synthetic audio (sine wave)."""
    t = np.linspace(0, 1.0, _SAMPLE_RATE, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)


# Re-import constants for test use.
from minty_box.stt import (
    _BLOCK_SIZE,
    _CHANNELS,
    _DEFAULT_SILENCE_DURATION,
    _DEFAULT_SILENCE_THRESHOLD,
    _DEFAULT_TIMEOUT,
    _DTYPE,
    _RESPEAKER_NAME,
    _SAMPLE_RATE,
)


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
        """Default silence params match module constants."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText()
            assert stt._silence_threshold == _DEFAULT_SILENCE_THRESHOLD
            assert stt._silence_duration == _DEFAULT_SILENCE_DURATION

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


# ---------------------------------------------------------------------------
# transcribe_buffer
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
            stt = SpeechToText()
            result = stt.transcribe_buffer(speech_audio)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert result.language == "en"
        assert result.language_probability == 0.99
        assert len(result.segments) == 1

    def test_empty_buffer_raises(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Empty audio raises ValueError."""
        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText()
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
            stt = SpeechToText()
            audio = np.array([16384, -16384, 0], dtype=np.int16)
            stt.transcribe_buffer(audio)

        # Inspect what was passed to model.transcribe.
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
        seg2 = MagicMock()
        seg2.text = "world"
        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        info.duration = 3.0
        mock_whisper_model.transcribe.return_value = ([seg1, seg2], info)

        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText()
            result = stt.transcribe_buffer(speech_audio)

        assert result.text == "hello world"
        assert result.segments == ["hello", "world"]

    def test_empty_segments_yields_empty_text(
        self, mock_whisper_model: MagicMock, speech_audio: np.ndarray,
    ) -> None:
        """Transcription with no segments returns empty string."""
        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        info.duration = 1.0
        mock_whisper_model.transcribe.return_value = ([], info)

        with (
            patch("minty_box.stt.WhisperModel", return_value=mock_whisper_model),
            patch("minty_box.stt.sd.query_devices", return_value=[]),
        ):
            stt = SpeechToText()
            result = stt.transcribe_buffer(speech_audio)

        assert result.text == ""
        assert result.segments == []


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
            )

    def test_stops_on_silence(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Recording ends when silence is detected."""
        stt = self._make_stt(mock_whisper_model, silence_duration=0.16)  # 2 frames

        call_count = [0]

        def fake_stream(**kwargs):
            """Mock InputStream that returns loud then silent frames."""
            class FakeStream:
                def __enter__(self2):
                    return self2
                def __exit__(self2, *args):
                    pass
                def read(self2, frames):
                    call_count[0] += 1
                    if call_count[0] <= 10:
                        # Loud frames for ~0.8s (passes 0.5s minimum speech guard).
                        return np.full((_BLOCK_SIZE, 1), 16384, dtype=np.int16), False
                    else:
                        # Then silent — should trigger endpoint after 2 quiet frames.
                        return np.zeros((_BLOCK_SIZE, 1), dtype=np.int16), False
            return FakeStream()

        with patch("minty_box.stt.sd.InputStream", side_effect=fake_stream):
            result = stt.listen_and_transcribe(timeout=2.0)

        # Should stop on silence, well before the 2.0s timeout.
        max_frames = int(2.0 * _SAMPLE_RATE / _BLOCK_SIZE)
        assert call_count[0] < max_frames
        assert isinstance(result, TranscriptionResult)

    def test_requires_minimum_speech_before_silence(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Continuous silence at start doesn't immediately exit."""
        stt = self._make_stt(mock_whisper_model, silence_duration=0.08)  # 1 frame

        call_count = [0]

        def fake_stream(**kwargs):
            class FakeStream:
                def __enter__(self2):
                    return self2
                def __exit__(self2, *args):
                    pass
                def read(self2, frames):
                    call_count[0] += 1
                    if call_count[0] <= 7:
                        return np.zeros((_BLOCK_SIZE, 1), dtype=np.int16), False
                    # After 0.5s of silence, play a loud frame.
                    return np.full((_BLOCK_SIZE, 1), 16384, dtype=np.int16), False
            return FakeStream()

        with patch("minty_box.stt.sd.InputStream", side_effect=fake_stream):
            result = stt.listen_and_transcribe(timeout=5.0)

        # Should not exit; hits timeout after 5s.
        assert call_count[0] >= 7
        assert isinstance(result, TranscriptionResult)

    def test_timeout_ends_recording(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """Recording ends at timeout even without silence."""
        stt = self._make_stt(mock_whisper_model)

        def fake_stream(**kwargs):
            class FakeStream:
                def __enter__(self2):
                    return self2
                def __exit__(self2, *args):
                    pass
                def read(self2, frames):
                    # Always loud — never silent.
                    return np.full((_BLOCK_SIZE, 1), 16384, dtype=np.int16), False
            return FakeStream()

        start = time.monotonic()
        with patch("minty_box.stt.sd.InputStream", side_effect=fake_stream):
            result = stt.listen_and_transcribe(timeout=1.0)
        elapsed = time.monotonic() - start

        assert elapsed < 1.5  # should be close to 1.0s
        assert isinstance(result, TranscriptionResult)

    def test_no_audio_returns_empty(
        self, mock_whisper_model: MagicMock,
    ) -> None:
        """If no frames recorded, result.text is ''."""
        stt = self._make_stt(mock_whisper_model)

        # Patch transcribe_buffer to raise if called (shouldn't be).
        original = stt.transcribe_buffer
        stt.transcribe_buffer = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError("should not be called")
        )

        def fake_stream(**kwargs):
            class FakeStream:
                def __enter__(self2):
                    return self2
                def __exit__(self2, *args):
                    pass
                def read(self2, frames):
                    raise KeyboardInterrupt  # immediate abort
            return FakeStream()

        with patch("minty_box.stt.sd.InputStream", side_effect=fake_stream):
            result = stt.listen_and_transcribe(timeout=0.01)

        assert result.text == ""

        # Restore.
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
            stt = SpeechToText()
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
        assert result.segments == []
        assert result.language == ""
        assert result.language_probability == 0.0
        assert result.duration == 0.0

    def test_full_construction(self) -> None:
        """All fields can be set explicitly."""
        result = TranscriptionResult(
            text="hello world",
            segments=["hello", "world"],
            language="en",
            language_probability=0.99,
            duration=2.5,
        )
        assert result.text == "hello world"
        assert result.segments == ["hello", "world"]
        assert result.language == "en"
        assert result.language_probability == 0.99
        assert result.duration == 2.5
