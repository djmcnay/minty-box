"""Tests for minty_box.tts — KokoroTTS."""

from __future__ import annotations

from unittest import mock

import pytest

from minty_box.tts import KokoroTTS


class MockHTTPResponse:
    """A minimal urllib-style response for testing."""

    def __init__(self, status=200, body=b"WAVDATA"):
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestKokoroTTS:
    def test_synthesis_returns_wav_bytes(self):
        tts = KokoroTTS()
        with mock.patch(
            "urllib.request.urlopen",
            return_value=MockHTTPResponse(body=b"FAKEWAV\x00\xff"),
        ):
            result = tts.synthesize("Hello")
        assert isinstance(result, bytes)
        assert result == b"FAKEWAV\x00\xff"

    def test_non_200_status_raises_valueerror(self):
        tts = KokoroTTS()
        with mock.patch(
            "urllib.request.urlopen",
            return_value=MockHTTPResponse(status=500, body=b"Server Error"),
        ):
            with pytest.raises(ValueError, match="500"):
                tts.synthesize("Hello")

    def test_connection_refused_raises_urlerror(self):
        tts = KokoroTTS()
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=OSError("Connection refused"),
        ):
            with pytest.raises(OSError):
                tts.synthesize("Hello")

    def test_custom_voice_and_speed_stored(self):
        tts = KokoroTTS(voice="af_heart", speed=1.0)
        assert tts._voice == "af_heart"
        assert tts._speed == 1.0

    def test_empty_text_produces_audio(self):
        """Even empty text should produce (silent) audio."""
        tts = KokoroTTS()
        with mock.patch(
            "urllib.request.urlopen",
            return_value=MockHTTPResponse(body=b"SILENT"),
        ):
            result = tts.synthesize("")
        assert isinstance(result, bytes)
        assert result == b"SILENT"
