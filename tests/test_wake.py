"""Tests for minty_box."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from minty_box import WakeWordListener
from minty_box.wake import (
    _BLOCK_SIZE,
    _BUILTIN_MODELS,
    _COOLDOWN_SECONDS,
    _DEFAULT_THRESHOLD,
    _SAMPLE_RATE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model() -> MagicMock:
    """Return a mock openWakeWord Model that returns fixed predictions."""
    model = MagicMock()
    model.predict.return_value = {"alexa": 0.0}
    model.models = {"alexa": MagicMock()}
    return model


@pytest.fixture
def mock_sd() -> MagicMock:
    """Mock sounddevice with a fake ReSpeaker Lite in the device list."""
    sd_mock = MagicMock()
    sd_mock.query_devices.return_value = [
        {"name": "pulse", "max_input_channels": 2},
        {"name": "ReSpeaker Lite: USB Audio (hw:2,0)", "max_input_channels": 2},
    ]
    sd_mock.default.device = [0]
    return sd_mock


@pytest.fixture
def on_wake() -> MagicMock:
    """Spy callback for wake word detections."""
    return MagicMock()


# ---------------------------------------------------------------------------
# _speex_available
# ---------------------------------------------------------------------------

class TestSpeexAvailable:
    """Tests for :meth:`WakeWordListener._speex_available`."""

    def test_returns_false_when_module_missing(
        self, on_wake: MagicMock, mock_model: MagicMock, mock_sd: MagicMock
    ) -> None:
        """On ARM64 / systems without the speexdsp_ns wheel."""
        with (
            patch("minty_box.wake.Model", return_value=mock_model),
            patch("minty_box.wake.sd", mock_sd),
        ):
            listener = WakeWordListener(on_wake=on_wake)
            assert listener._speex_available() is False

    def test_returns_true_when_module_present(
        self, on_wake: MagicMock, mock_model: MagicMock, mock_sd: MagicMock
    ) -> None:
        """If the x86-only wheel is somehow available."""
        with (
            patch("minty_box.wake.Model", return_value=mock_model),
            patch("minty_box.wake.sd", mock_sd),
            patch.dict(
                "sys.modules",
                {"speexdsp_ns": MagicMock()},
            ),
        ):
            listener = WakeWordListener(on_wake=on_wake)
            assert listener._speex_available() is True


# ---------------------------------------------------------------------------
# _find_respeaker
# ---------------------------------------------------------------------------

class TestFindRespeaker:
    """Tests for :meth:`WakeWordListener._find_respeaker`."""

    def test_finds_respeaker_by_name(
        self, on_wake: MagicMock, mock_model: MagicMock, mock_sd: MagicMock
    ) -> None:
        """Returns the index of the device whose name contains 'ReSpeaker Lite'."""
        with (
            patch("minty_box.wake.Model", return_value=mock_model),
            patch("minty_box.wake.sd", mock_sd),
        ):
            listener = WakeWordListener(on_wake=on_wake)
            assert listener._find_respeaker() == 1  # second in list

    def test_fallback_when_respeaker_absent(
        self, on_wake: MagicMock, mock_model: MagicMock, mock_sd: MagicMock
    ) -> None:
        """Uses system default when no ReSpeaker Lite is found."""
        mock_sd.query_devices.return_value = [
            {"name": "pulse", "max_input_channels": 2},
        ]
        mock_sd.default.device = [0]

        with (
            patch("minty_box.wake.Model", return_value=mock_model),
            patch("minty_box.wake.sd", mock_sd),
        ):
            listener = WakeWordListener(on_wake=on_wake)
            assert listener._find_respeaker() == 0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    """Tests for :class:`WakeWordListener` initialisation."""

    def test_builtin_models_when_no_paths_given(
        self, on_wake: MagicMock, mock_model: MagicMock, mock_sd: MagicMock
    ) -> None:
        """Loads all built-in models when ``model_paths`` is ``None``."""
        with (
            patch("minty_box.wake.Model", return_value=mock_model),
            patch("minty_box.wake.sd", mock_sd),
        ):
            listener = WakeWordListener(on_wake=on_wake)
            assert listener._model_names == _BUILTIN_MODELS

    def test_custom_model_paths(
        self, on_wake: MagicMock, mock_model: MagicMock, mock_sd: MagicMock
    ) -> None:
        """Uses only the specified model when paths are given."""
        with (
            patch("minty_box.wake.Model", return_value=mock_model),
            patch("minty_box.wake.sd", mock_sd),
        ):
            listener = WakeWordListener(
                on_wake=on_wake,
                model_paths=["models/araminta.onnx"],
            )
            assert listener._model_names == ["araminta"]

    def test_custom_threshold(
        self, on_wake: MagicMock, mock_model: MagicMock, mock_sd: MagicMock
    ) -> None:
        """Threshold is stored as given."""
        with (
            patch("minty_box.wake.Model", return_value=mock_model),
            patch("minty_box.wake.sd", mock_sd),
        ):
            listener = WakeWordListener(on_wake=on_wake, threshold=0.7)
            assert listener._threshold == 0.7

    def test_default_threshold(
        self, on_wake: MagicMock, mock_model: MagicMock, mock_sd: MagicMock
    ) -> None:
        """When no threshold is given, uses ``_DEFAULT_THRESHOLD``."""
        with (
            patch("minty_box.wake.Model", return_value=mock_model),
            patch("minty_box.wake.sd", mock_sd),
        ):
            listener = WakeWordListener(on_wake=on_wake)
            assert listener._threshold == _DEFAULT_THRESHOLD

    def test_default_cooldown(
        self, on_wake: MagicMock, mock_model: MagicMock, mock_sd: MagicMock
    ) -> None:
        """Default cooldown is ``_COOLDOWN_SECONDS``."""
        with (
            patch("minty_box.wake.Model", return_value=mock_model),
            patch("minty_box.wake.sd", mock_sd),
        ):
            listener = WakeWordListener(on_wake=on_wake)
            assert listener._cooldown == _COOLDOWN_SECONDS

    def test_custom_device(
        self, on_wake: MagicMock, mock_model: MagicMock, mock_sd: MagicMock
    ) -> None:
        """Respects explicit device string."""
        with (
            patch("minty_box.wake.Model", return_value=mock_model),
            patch("minty_box.wake.sd", mock_sd),
        ):
            listener = WakeWordListener(
                on_wake=on_wake,
                input_device="pulse",
            )
            assert listener._device == "pulse"

    def test_running_starts_false(
        self, on_wake: MagicMock, mock_model: MagicMock, mock_sd: MagicMock
    ) -> None:
        """``_running`` is ``False`` after construction."""
        with (
            patch("minty_box.wake.Model", return_value=mock_model),
            patch("minty_box.wake.sd", mock_sd),
        ):
            listener = WakeWordListener(on_wake=on_wake)
            assert listener._running is False


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------

class TestStop:
    """Tests for :meth:`WakeWordListener.stop`."""

    def test_stop_sets_running_false(
        self, on_wake: MagicMock, mock_model: MagicMock, mock_sd: MagicMock
    ) -> None:
        """After calling stop(), _running is False."""
        with (
            patch("minty_box.wake.Model", return_value=mock_model),
            patch("minty_box.wake.sd", mock_sd),
        ):
            listener = WakeWordListener(on_wake=on_wake)
            listener._running = True
            listener.stop()
            assert listener._running is False


# ---------------------------------------------------------------------------
# _audio_callback
# ---------------------------------------------------------------------------

class TestAudioCallback:
    """Tests for :meth:`WakeWordListener._audio_callback`."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fake_audio(frames: int = 1280) -> np.ndarray:
        """Return a block of silent audio."""
        return np.zeros((frames, 1), dtype=np.int16)

    @staticmethod
    def _make_listener(
        on_wake: MagicMock,
        threshold: float = _DEFAULT_THRESHOLD,
        cooldown: float = _COOLDOWN_SECONDS,
        enable_vad: bool = True,
    ) -> WakeWordListener:
        """Build a listener with a mock model and controlled params."""
        model = MagicMock()
        model.predict.return_value = {}

        with patch("minty_box.wake.sd.query_devices", return_value=[]):
            listener = WakeWordListener(
                on_wake=on_wake,
                threshold=threshold,
                cooldown=cooldown,
                enable_vad=enable_vad,
                input_device=0,
            )
        # Replace the model with the one we control.
        listener._model = model
        return listener

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_calls_on_wake_when_score_above_threshold(self, on_wake: MagicMock) -> None:
        """Callback fires when a model's score exceeds threshold."""
        listener = self._make_listener(on_wake, threshold=0.5)
        listener._model.predict.return_value = {"alexa": 0.9}

        listener._audio_callback(
            self._fake_audio(), 1280, MagicMock(), 0,
        )

        on_wake.assert_called_once_with("alexa", 0.9)

    def test_does_not_call_on_wake_below_threshold(self, on_wake: MagicMock) -> None:
        """No callback when score is below the threshold."""
        listener = self._make_listener(on_wake, threshold=0.5)
        listener._model.predict.return_value = {"alexa": 0.4}

        listener._audio_callback(
            self._fake_audio(), 1280, MagicMock(), 0,
        )

        on_wake.assert_not_called()

    def test_cooldown_suppresses_repeated_detections(
        self, on_wake: MagicMock,
    ) -> None:
        """Second detection within cooldown window is ignored."""
        listener = self._make_listener(on_wake, cooldown=2.0)
        listener._model.predict.return_value = {"alexa": 0.9}

        # First call fires.
        listener._audio_callback(
            self._fake_audio(), 1280, MagicMock(), 0,
        )
        assert on_wake.call_count == 1

        # Immediate second call — suppressed.
        listener._audio_callback(
            self._fake_audio(), 1280, MagicMock(), 0,
        )
        assert on_wake.call_count == 1  # still 1

    def test_cooldown_allows_after_window(
        self, on_wake: MagicMock,
    ) -> None:
        """Detection fires again after the cooldown expires."""
        listener = self._make_listener(on_wake, cooldown=0.1)
        listener._model.predict.return_value = {"alexa": 0.9}

        # First.
        listener._audio_callback(
            self._fake_audio(), 1280, MagicMock(), 0,
        )
        # Wait past cooldown.
        time.sleep(0.15)
        # Second.
        listener._audio_callback(
            self._fake_audio(), 1280, MagicMock(), 0,
        )

        assert on_wake.call_count == 2

    def test_different_models_have_independent_cooldown(
        self, on_wake: MagicMock,
    ) -> None:
        """Two models can fire independently even within cooldown."""
        listener = self._make_listener(on_wake, cooldown=5.0)
        listener._model.predict.return_value = {
            "alexa": 0.9,
            "hey_jarvis": 0.9,
        }

        listener._audio_callback(
            self._fake_audio(), 1280, MagicMock(), 0,
        )

        assert on_wake.call_count == 2
        calls = [call.args[0] for call in on_wake.call_args_list]
        assert "alexa" in calls
        assert "hey_jarvis" in calls

    def test_callback_exception_is_logged_not_raised(
        self, on_wake: MagicMock,
    ) -> None:
        """If on_wake raises, the error is logged and the listener continues."""
        on_wake.side_effect = RuntimeError("boom")
        listener = self._make_listener(on_wake)
        listener._model.predict.return_value = {"alexa": 0.9}

        # Must not propagate.
        listener._audio_callback(
            self._fake_audio(), 1280, MagicMock(), 0,
        )
        # on_wake was called (and raised inside).
        on_wake.assert_called_once()

    def test_handles_multi_channel_input(self, on_wake: MagicMock) -> None:
        """Collapses multi-channel audio to 1-D before model.predict."""
        listener = self._make_listener(on_wake, threshold=0.5)
        listener._model.predict.return_value = {"alexa": 0.6}

        # 2-channel input — callback should extract channel 0.
        audio_2ch = np.zeros((1280, 2), dtype=np.int16)
        listener._audio_callback(audio_2ch, 1280, MagicMock(), 0)

        # Verify the model received 1-D audio.
        called_audio = listener._model.predict.call_args[0][0]
        assert called_audio.ndim == 1
        assert called_audio.shape == (1280,)
        on_wake.assert_called_once()

    def test_multi_model_predictions_filtered_independently(
        self, on_wake: MagicMock,
    ) -> None:
        """Models below threshold are skipped while those above fire."""
        listener = self._make_listener(on_wake, threshold=0.5)
        listener._model.predict.return_value = {
            "alexa": 0.9,
            "hey_jarvis": 0.3,
            "timer": 0.8,
        }

        listener._audio_callback(
            self._fake_audio(), 1280, MagicMock(), 0,
        )

        assert on_wake.call_count == 2
        called_models = {call.args[0] for call in on_wake.call_args_list}
        assert called_models == {"alexa", "timer"}


# ---------------------------------------------------------------------------
# Utterance capture
# ---------------------------------------------------------------------------

class TestUtteranceCapture:
    """Tests for the built-in utterance capture pipeline.

    Verifies that after a wake word fires (with ``on_utterance``
    registered) the listener enters capture mode, accumulates frames,
    detects silence, and emits the buffer without opening a second
    stream.
    """

    @staticmethod
    def _make_listener(
        on_wake: MagicMock,
        on_utterance: MagicMock | None = None,
        **kwargs: Any,
    ) -> WakeWordListener:
        """Build a ``WakeWordListener`` with a mocked Model, controlled
        capture params, and a fake input device."""
        model = MagicMock()
        model.predict.return_value = {}

        with patch("minty_box.wake.sd.query_devices", return_value=[]):
            kwargs.setdefault("capture_silence_duration", 0.16)
            kwargs.setdefault("capture_timeout", 5.0)
            listener = WakeWordListener(
                on_wake=on_wake,
                on_utterance=on_utterance,
                input_device=0,
                **kwargs,
            )
        listener._model = model
        return listener

    @staticmethod
    def _loud_frame() -> np.ndarray:
        """A frame well above the silence threshold."""
        return np.full((_BLOCK_SIZE, 1), 16384, dtype=np.int16)

    @staticmethod
    def _silent_frame() -> np.ndarray:
        """A frame well below the silence threshold."""
        return np.zeros((_BLOCK_SIZE, 1), dtype=np.int16)

    def test_enters_capture_after_wake_word(self, on_wake: MagicMock) -> None:
        """When a detection fires and on_utterance is set, the listener
        enters capture mode."""
        on_utterance = MagicMock()
        listener = self._make_listener(on_wake, on_utterance)
        listener._model.predict.return_value = {"hey_jarvis": 0.9}

        listener._audio_callback(
            self._loud_frame(), _BLOCK_SIZE, MagicMock(), 0,
        )

        on_wake.assert_called_once()
        assert listener._capturing is True

    def test_does_not_enter_capture_without_on_utterance(
        self, on_wake: MagicMock,
    ) -> None:
        """No capture is attempted when on_utterance is None."""
        listener = self._make_listener(on_wake, on_utterance=None)
        listener._model.predict.return_value = {"hey_jarvis": 0.9}

        listener._audio_callback(
            self._loud_frame(), _BLOCK_SIZE, MagicMock(), 0,
        )

        on_wake.assert_called_once()
        assert listener._capturing is False

    def test_captures_and_emits_on_silence(self, on_wake: MagicMock) -> None:
        """After entering capture, the listener accumulates frames until
        the silence window is full, then calls on_utterance."""
        on_utterance = MagicMock()
        listener = self._make_listener(on_wake, on_utterance)
        listener._model.predict.return_value = {"hey_jarvis": 0.9}

        # First frame triggers detection → enter capture mode.
        listener._audio_callback(
            self._loud_frame(), _BLOCK_SIZE, MagicMock(), 0,
        )
        on_wake.assert_called_once()
        assert listener._capturing is True

        # Feed loud speech frames (need > min_speech_frames = 7).
        for _ in range(8):
            listener._audio_callback(
                self._loud_frame(), _BLOCK_SIZE, MagicMock(), 0,
            )

        # Feed two silent frames to fill the silence window.
        listener._audio_callback(
            self._silent_frame(), _BLOCK_SIZE, MagicMock(), 0,
        )
        listener._audio_callback(
            self._silent_frame(), _BLOCK_SIZE, MagicMock(), 0,
        )

        # on_utterance should have been called exactly once.
        on_utterance.assert_called_once()
        emitted_audio = on_utterance.call_args[0][0]
        assert isinstance(emitted_audio, np.ndarray)
        assert emitted_audio.dtype == np.int16
        # Should have accumulated: 8 loud + 2 silent = 10 frames.
        assert emitted_audio.shape[0] == _BLOCK_SIZE * 10

        # Listener should have returned to listening mode.
        assert listener._capturing is False

    def test_timeout_emits_utterance(self, on_wake: MagicMock) -> None:
        """When the stream is constantly loud, the capture timeout ends
        the recording and emits the buffer."""
        on_utterance = MagicMock()
        # Short timeout = 4 frames.
        listener = self._make_listener(
            on_wake, on_utterance, capture_timeout=4 * _BLOCK_SIZE / _SAMPLE_RATE,
        )
        listener._model.predict.return_value = {"hey_jarvis": 0.9}

        # Trigger.
        listener._audio_callback(
            self._loud_frame(), _BLOCK_SIZE, MagicMock(), 0,
        )

        # Feed constantly loud — enough to hit max_frames=4.
        for _ in range(4):
            listener._audio_callback(
                self._loud_frame(), _BLOCK_SIZE, MagicMock(), 0,
            )

        on_utterance.assert_called_once()
        assert listener._capturing is False

    def test_no_utterance_callback_emitted_without_min_speech(
        self, on_wake: MagicMock,
    ) -> None:
        """Silence immediately after wake word doesn't emit until
        minimum speech frames have been recorded."""
        on_utterance = MagicMock()
        listener = self._make_listener(on_wake, on_utterance)
        listener._model.predict.return_value = {"hey_jarvis": 0.9}

        # Trigger.
        listener._audio_callback(
            self._loud_frame(), _BLOCK_SIZE, MagicMock(), 0,
        )

        # Feed only 3 frames before silence — should NOT emit.
        for _ in range(3):
            listener._audio_callback(
                self._loud_frame(), _BLOCK_SIZE, MagicMock(), 0,
            )
        # Then silence.
        listener._audio_callback(
            self._silent_frame(), _BLOCK_SIZE, MagicMock(), 0,
        )
        listener._audio_callback(
            self._silent_frame(), _BLOCK_SIZE, MagicMock(), 0,
        )

        # Still not emitted (min_speech_frames = 7, we only had 3 loud).
        on_utterance.assert_not_called()
        assert listener._capturing is True

    def test_wake_word_break_on_capture(self, on_wake: MagicMock) -> None:
        """When capture mode is entered, only the first detection in
        a frame fires.  Multi-model detections in the same frame are
        suppressed to avoid capture-state corruption."""
        on_utterance = MagicMock()
        listener = self._make_listener(on_wake, on_utterance)
        listener._model.predict.return_value = {
            "alexa": 0.9,
            "hey_jarvis": 0.9,
            "timer": 0.9,
        }

        listener._audio_callback(
            self._loud_frame(), _BLOCK_SIZE, MagicMock(), 0,
        )

        # Only one wake callback, not three.
        assert on_wake.call_count == 1
        assert listener._capturing is True


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLI:
    """Tests for :func:`main` argument parsing (side-effect free paths)."""

    def test_invalid_model_path_rejected(self) -> None:
        """argparse exits with error for non-existent model file."""
        import argparse
        import sys

        from minty_box.wake import main as cli_main

        test_args = ["--model", "/nonexistent/path.onnx"]
        with (
            patch.object(sys, "argv", ["minty_box.wake"] + test_args),
            pytest.raises(SystemExit),
        ):
            cli_main()


# ---------------------------------------------------------------------------
# Integration-style: full construction path
# ---------------------------------------------------------------------------

class TestIntegration:
    """Verify the listener can be constructed end-to-end with mocks."""

    def test_listener_constructs_with_real_model_module(
        self, on_wake: MagicMock,
    ) -> None:
        """Smoke test: instantiates WakeWordListener with the actual
        openWakeWord Model class (no mocks on the Model itself).
        Requires openWakeWord to be installed.
        """
        with patch("minty_box.wake.sd") as mock_sd:
            mock_sd.query_devices.return_value = [
                {"name": "ReSpeaker Lite: USB Audio (hw:2,0)"},
            ]

            listener = WakeWordListener(
                on_wake=on_wake,
                threshold=0.5,
                enable_vad=False,
            )

            assert isinstance(listener._model_names, list)
            assert len(listener._model_names) > 0
            assert listener._threshold == 0.5
