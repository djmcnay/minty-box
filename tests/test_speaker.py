"""Tests for minty_box.speaker — Speaker and _generate_sine_wav."""

from __future__ import annotations

import subprocess
import wave
from unittest import mock

import pytest

from minty_box.speaker import Speaker, _generate_sine_wav


class TestGenerateSineWav:
    def test_produces_valid_wav(self):
        wav_bytes = _generate_sine_wav(330, 0.15)
        # Should be a valid WAV we can read back.
        import io
        with wave.open(io.BytesIO(wav_bytes), "r") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            # 0.15s at 16 kHz = 2400 samples
            assert wf.getnframes() == 2400

    def test_different_duration(self):
        wav_bytes = _generate_sine_wav(440, 0.5)
        import io
        with wave.open(io.BytesIO(wav_bytes), "r") as wf:
            assert wf.getnframes() == 8000  # 0.5 * 16000


class TestSpeaker:
    def test_plays_wav_bytes(self):
        speaker = Speaker()
        with mock.patch("subprocess.run") as mock_run:
            speaker.play(_generate_sine_wav(440, 0.1), volume=0.08)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]  # the command list
        assert args[0] == "ffplay"
        assert "-nodisp" in args
        assert "-autoexit" in args
        assert any("volume=0.08" in a for a in args)

    def test_ffplay_failure_raises(self):
        speaker = Speaker()
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(
                1, "ffplay", stderr="No such device"
            ),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                speaker.play(_generate_sine_wav(440, 0.1))

    def test_temp_file_cleaned_up_on_error(self):
        speaker = Speaker()
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffplay"),
        ), mock.patch("pathlib.Path.unlink") as mock_unlink:
            with pytest.raises(subprocess.CalledProcessError):
                speaker.play(_generate_sine_wav(440, 0.1))
        mock_unlink.assert_called()

    def test_beep_calls_ffplay(self):
        speaker = Speaker()
        with mock.patch("subprocess.run"):
            speaker.beep()
        # Should not raise — beep swallows failures.

    def test_beep_does_not_raise_on_ffplay_failure(self):
        """beep() must never propagate exceptions — it's cosmetic."""
        speaker = Speaker()
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffplay"),
        ):
            speaker.beep()  # no exception
