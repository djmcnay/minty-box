"""Text-to-speech via the local Kokoro ONNX HTTP service.

The Kokoro server (``~/.hermes/kokoro-tts/server.py``) must be running
on ``localhost:8880`` before any synthesis calls.  Start it with:

    cd ~/.hermes/kokoro-tts && ./venv/bin/python server.py
"""

from __future__ import annotations

import json
import logging
import urllib.request
from urllib.error import URLError

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "bf_emma"
DEFAULT_SPEED = 1.15  # +15% — Emma sounds natural here
DEFAULT_TIMEOUT = 30
KOKORO_URL = "http://127.0.0.1:8880/v1/audio/speech"


class KokoroTTS:
    """Synthesise speech via the local Kokoro HTTP API.

    The server exposes an OpenAI-compatible ``/v1/audio/speech``
    endpoint.  Synthesis is synchronous — it blocks until the full
    WAV is returned by the server.

    Output is 24 kHz 16-bit PCM WAV.
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """Args:
            voice: Kokoro voice name (``bf_emma`` default — British female).
            speed: Playback speed multiplier (1.0 = normal, 1.15 default).
            timeout: HTTP request timeout in seconds.
        """
        self._voice = voice
        self._speed = speed
        self._timeout = timeout

    def synthesize(self, text: str) -> bytes:
        """Convert *text* to WAV audio bytes.

        Args:
            text: The text to synthesise (any length; Kokoro handles
                punctuation for prosody).

        Returns:
            PCM 16-bit WAV audio at 24 kHz.

        Raises:
            URLError: If the Kokoro server is unreachable (connection
                refused, timeout, DNS failure).
            ValueError: If the server returns a non-200 status.
        """
        payload = json.dumps(
            {
                "model": "kokoro",
                "input": text,
                "voice": self._voice,
                "response_format": "wav",
                "speed": self._speed,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            KOKORO_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status != 200:
                    body = resp.read().decode(errors="replace")
                    raise ValueError(
                        f"Kokoro returned {resp.status}: {body[:200]}"
                    )
                wav_bytes = resp.read()
        except URLError as e:
            logger.error("Kokoro unreachable at %s: %s", KOKORO_URL, e)
            raise

        logger.info(
            "Synthesised %d chars → %d bytes WAV", len(text), len(wav_bytes)
        )
        return wav_bytes
