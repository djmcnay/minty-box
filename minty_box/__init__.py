"""Minty Box — Araminta's physical hardware companion.

A Raspberry Pi 5 voice interface with ReSpeaker Lite mic array, speaker,
and touch display. Runs entirely offline.

Exports:
    WakeWordListener: continuous wake word detector using openWakeWord.
"""

from minty_box.wake import WakeWordListener

__all__ = ["WakeWordListener"]
