"""Audio playback through the ReSpeaker Lite speaker.

Uses PulseAudio routing — the ReSpeaker sink should already be
configured as the default ALSA output.  Playback goes through
``ffplay`` with no explicit device, letting PulseAudio handle
sink selection.
"""

from __future__ import annotations

import logging
import math
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_VOLUME = 0.08  # daytime normal per calibration table
BEEP_FREQ = 330  # Hz — low, dulcet
BEEP_DURATION = 0.15  # seconds
BEEP_VOLUME = 0.03  # subtle acknowledgment
SAMPLE_RATE = 16000
PLAYBACK_TIMEOUT = 30


class Speaker:
    """Plays WAV audio through the ReSpeaker Lite speaker.

    Writes audio to a temporary ``.wav`` file, plays it via
    ``ffplay`` in headless mode (no video window), and cleans
    up the temp file afterwards — even on error.
    """

    def play(self, wav_bytes: bytes, volume: float = DEFAULT_VOLUME) -> None:
        """Play *wav_bytes* through the ReSpeaker speaker.

        Args:
            wav_bytes: PCM WAV audio data.
            volume: Linear amplitude scale.  Reference values:
                ``0.01`` whisper, ``0.02`` quiet night, ``0.08``
                daytime normal, ``0.20`` loud, ``1.0`` maximum.

        Raises:
            subprocess.CalledProcessError: If ``ffplay`` exits
                non-zero (device missing, format unsupported).
        """
        tmp_path = self._write_temp(wav_bytes)
        try:
            self._ffplay(tmp_path, volume)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def beep(self) -> None:
        """Play a short, low-pitched acknowledgment tone.

        ~330 Hz for 150 ms at low volume — subtle enough to avoid
        being intrusive, distinctive enough to confirm the wake
        word was heard.
        """
        wav_bytes = _generate_sine_wav(BEEP_FREQ, BEEP_DURATION)
        tmp_path = self._write_temp(wav_bytes)
        try:
            self._ffplay(tmp_path, BEEP_VOLUME)
        except subprocess.CalledProcessError:
            logger.warning("Beep playback failed — continuing")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── internal helpers ───────────────────────────────────────────

    @staticmethod
    def _write_temp(data: bytes) -> str:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as f:
            f.write(data)
            return f.name

    @staticmethod
    def _ffplay(path: str, volume: float) -> None:
        subprocess.run(
            [
                "ffplay", "-nodisp", "-autoexit",
                "-af", f"volume={volume}",
                path,
            ],
            check=True,
            timeout=PLAYBACK_TIMEOUT,
            capture_output=True,
            text=True,
        )


def _generate_sine_wav(freq: float, duration: float) -> bytes:
    """Generate a mono 16-bit PCM WAV of a sine tone.

    Args:
        freq: Frequency in Hz.
        duration: Duration in seconds.

    Returns:
        Complete WAV file as bytes (16 kHz, mono, 16-bit).
    """
    n_samples = int(SAMPLE_RATE * duration)
    samples = [
        int(16000 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        for i in range(n_samples)
    ]

    buf = io_bytes()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return buf.getvalue()


# Python 3.14 still uses io.BytesIO — this is just a readability alias.
from io import BytesIO as _BytesIO


def io_bytes() -> _BytesIO:
    return _BytesIO()
