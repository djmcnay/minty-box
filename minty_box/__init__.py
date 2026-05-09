"""Minty Box — Araminta's physical hardware companion.

A Raspberry Pi 5 voice interface with ReSpeaker Lite mic array, speaker,
and touch display. Runs entirely offline.

Exports:
    WakeWordListener: continuous wake word detector using openWakeWord.
    SpeechToText: speech-to-text transcription via faster-whisper.
    TranscriptionResult: dataclass holding STT output.
"""

from minty_box.stt import SpeechToText, TranscriptionResult
from minty_box.wake import WakeWordListener

__all__ = [
    "SpeechToText",
    "TranscriptionResult",
    "WakeWordListener",
]
