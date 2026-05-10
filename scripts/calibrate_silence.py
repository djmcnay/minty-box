"""
Ambient noise calibration for the Minty Box silence detector.

Records audio from the ReSpeaker Lite while the room is quiet
(occupants typing, shifting, breathing — normal background, just
no talking) and computes the RMS energy distribution.  From this
it recommends a silence threshold for `WakeWordListener`'s
utterance capture that sits safely above the ambient noise floor.

Usage:
    uv run python scripts/calibrate_silence.py           # 30s default
    uv run python scripts/calibrate_silence.py --duration 60
    uv run python scripts/calibrate_silence.py --device "ReSpeaker Lite"
    uv run python scripts/calibrate_silence.py --json /tmp/cal.json

Output:
    Prints summary statistics to stdout and, if `--json` is given,
    writes the full per-block RMS array for later inspection.

The recommended multiplier is deliberately conservative: we want
the silence threshold well above the 95th percentile so typing
and chair creaks don't prevent endpointing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000
BLOCK_SIZE: int = 1280  # 80 ms — matches wake.py's _BLOCK_SIZE
DTYPE: str = "int16"

# Multiplier applied to the median noise floor to derive the
# recommended silence threshold.  8× is conservative — it sits
# well above the 95th percentile in testing at Moorside Cottage.
_DEFAULT_MULTIPLIER: float = 8.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_device(name_hint: str | None = None) -> int | str:
    """Return the best input device for calibration.

    Priority: user-supplied name > ReSpeaker Lite by name > system default.
    """
    devices: list[dict] = sd.query_devices()

    if name_hint:
        for idx, dev in enumerate(devices):
            if (name_hint.lower() in dev["name"].lower()
                    and dev.get("max_input_channels", 0) > 0):
                return idx
        print(f"Device '{name_hint}' not found, falling back.", file=sys.stderr)

    # Auto-detect ReSpeaker Lite.
    for idx, dev in enumerate(devices):
        if ("ReSpeaker Lite" in dev["name"]
                and dev.get("max_input_channels", 0) > 0):
            return idx

    # Fallback: system default input.
    return sd.default.device[0]  # type: ignore[return-value]


def _compute_rms_blocks(audio: np.ndarray) -> np.ndarray:
    """Split audio into 80 ms blocks and compute normalised RMS for each.

    Parameters
    ----------
    audio:
        1-D int16 array at 16 kHz.

    Returns
    -------
    np.ndarray
        Float64 array of normalised RMS values (0–1), one per block.
    """
    n_blocks: int = len(audio) // BLOCK_SIZE
    audio = audio[: n_blocks * BLOCK_SIZE]  # drop incomplete tail
    blocks = audio.reshape(n_blocks, BLOCK_SIZE)
    # Normalised RMS: divide by 32768 (max int16) to get 0–1 range.
    rms = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1)) / 32768.0
    return rms


def _recommend(rms: np.ndarray, multiplier: float = _DEFAULT_MULTIPLIER) -> dict:
    """Compute statistics and a recommended silence threshold.

    Returns a dict with keys: min, mean, median, p90, p95, p99, max,
    noise_floor, recommended_threshold, multiplier.
    """
    noise_floor = float(np.median(rms))
    return {
        "min": float(rms.min()),
        "mean": float(rms.mean()),
        "median": float(np.median(rms)),
        "p90": float(np.percentile(rms, 90)),
        "p95": float(np.percentile(rms, 95)),
        "p99": float(np.percentile(rms, 99)),
        "max": float(rms.max()),
        "noise_floor": noise_floor,
        "recommended_threshold": round(noise_floor * multiplier, 4),
        "multiplier": multiplier,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Record ambient audio and recommend a silence threshold."""
    parser = argparse.ArgumentParser(
        description="Calibrate silence threshold for Minty Box wake word capture",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=30.0,
        help="Recording duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Input device name hint (auto-detects ReSpeaker Lite if omitted)",
    )
    parser.add_argument(
        "--multiplier", "-m",
        type=float,
        default=_DEFAULT_MULTIPLIER,
        help=f"Multiplier over noise floor for threshold (default: {_DEFAULT_MULTIPLIER})",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="If set, write full stats + per-block RMS array to this path",
    )
    args = parser.parse_args()

    # Resolve device.
    device = _find_device(args.device)
    dev_info = sd.query_devices(device)
    print(f"Device: {dev_info['name']}  (index {device})")
    print(f"Duration: {args.duration} s  |  Multiplier: {args.multiplier}×")
    print(f"Make some normal noise — typing, shifting, whatever.  Just don't talk.\n")

    # Record.
    n_samples = int(args.duration * SAMPLE_RATE)
    recording = sd.rec(
        n_samples,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=DTYPE,
        device=device,
        blocking=True,
    )
    audio: np.ndarray = recording[:, 0] if recording.ndim > 1 else recording

    # Analyse.
    rms = _compute_rms_blocks(audio)
    stats = _recommend(rms, multiplier=args.multiplier)

    # Report.
    print("─── Ambient RMS Distribution ───")
    print(f"  Min:            {stats['min']:>10.6f}")
    print(f"  Mean:           {stats['mean']:>10.6f}")
    print(f"  Median (floor): {stats['median']:>10.6f}")
    print(f"  90th %ile:      {stats['p90']:>10.6f}")
    print(f"  95th %ile:      {stats['p95']:>10.6f}")
    print(f"  99th %ile:      {stats['p99']:>10.6f}")
    print(f"  Max:            {stats['max']:>10.6f}")
    print()
    print("─── Recommendation ───")
    print(f"  Noise floor:             {stats['noise_floor']:.6f}")
    print(f"  Multiplier:              {stats['multiplier']}×")
    print(f"  Recommended threshold:   {stats['recommended_threshold']:.4f}")
    print()
    print("To apply, update _CAPTURE_SILENCE_THRESHOLD in minty_box/wake.py")

    # Optional JSON output.
    if args.json:
        output = {
            **stats,
            "rms_per_block": rms.tolist(),
            "device_name": dev_info["name"],
            "device_index": device,
            "duration_s": args.duration,
            "sample_rate": SAMPLE_RATE,
            "block_size": BLOCK_SIZE,
        }
        json_path = Path(args.json).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Full data written to {json_path}")


if __name__ == "__main__":
    main()
