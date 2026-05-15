"""Minty Box CLI launcher.

Process management, hardware checks, and graceful start/stop.
``minty-box start`` is the intended entry point — it scans for stale
processes, verifies hardware, warms up Kokoro, then launches the
listener as a background subprocess.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

STALE_PATTERNS = ["main.py", "minty-box", "minty_box"]
KOKORO_URL = "http://127.0.0.1:8880/health"


# ── public entry point ────────────────────────────────────────────────


def main() -> None:
    """CLI entry point wired into pyproject.toml ``[project.scripts]``."""
    parser = argparse.ArgumentParser(
        description="Minty Box — Araminta's voice companion launcher",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── start ────────────────────────────────────────────────────────
    p_start = sub.add_parser("start", help="Start the voice listener")
    p_start.add_argument(
        "--model",
        type=str,
        default="models/Araminta.onnx",
        help="Path to .onnx wake word model (default: models/Araminta.onnx)",
    )
    p_start.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Wake word detection threshold (default: 0.3)",
    )
    p_start.add_argument(
        "--warm",
        action="store_true",
        default=True,
        help="Use warm Hermes via tmux (default: on)",
    )
    p_start.add_argument(
        "--no-warm",
        action="store_true",
        help="Disable warm Hermes, use cold single-shot",
    )
    p_start.add_argument(
        "--no-captures",
        action="store_true",
        help="Don't save capture JSONs",
    )

    # ── stop ─────────────────────────────────────────────────────────
    sub.add_parser("stop", help="Stop the voice listener")

    # ── status ───────────────────────────────────────────────────────
    sub.add_parser("status", help="Check listener status")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "stop":
        cmd_stop()
    elif args.command == "status":
        cmd_status()


# ── command implementations ────────────────────────────────────────────


def cmd_start(args: argparse.Namespace) -> None:
    """Orchestrate a clean start: kill stale → check HW → launch."""
    print("📻 Minty Box — starting up...")
    print()

    # 1. Kill stale processes.
    killed = _find_and_kill_stale()
    if killed:
        print(f"   🧹 Killed {killed} stale process(es).")
    else:
        print("   ✓ No stale processes found.")

    # 2. Hardware check.
    if not _check_hardware():
        print("❌ ReSpeaker Lite not detected. Is it plugged in?")
        sys.exit(1)
    print("   🎙️  ReSpeaker Lite detected.")

    # 3. Kokoro check.
    if not _check_kokoro():
        print("⚠️  Kokoro TTS not responding — will use --no-tts mode.")
        print("   Start it: ~/.hermes/kokoro-tts/venv/bin/python server.py")
        tts_ok = False
    else:
        print("   🔊 Kokoro TTS ready.")
        tts_ok = True

    # 4. Warm Hermes session (if enabled).
    warm = args.warm and not args.no_warm
    if warm and tts_ok:
        from minty_box.session import WarmHermesSession
        session = WarmHermesSession()
        session.start()
        print("   🧠 Warm Hermes session ready.")
    else:
        if not warm:
            print("   🧠 Cold single-shot mode (no warm Hermes).")
        print()

    # 5. Launch listener.
    print()
    _start_listener(
        model=args.model,
        threshold=args.threshold,
        warm=warm,
        captures=not args.no_captures,
        no_tts=not tts_ok,
    )
    print("✅ Minty Box is listening.")
    print()


def cmd_stop() -> None:
    """Kill everything and verify."""
    killed = _find_and_kill_stale()
    if killed:
        print(f"🧹 Stopped {killed} process(es).")
    else:
        print("Nothing running.")
    # Also kill warm Hermes tmux session.
    from minty_box.session import WarmHermesSession
    WarmHermesSession().stop()
    print("✅ Minty Box stopped.")


def cmd_status() -> None:
    """Report whether the listener is running."""
    result = subprocess.run(
        ["pgrep", "-f", "main.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        pids = result.stdout.strip().split()
        print(f"🟢 Listening — PID(s): {', '.join(pids)}")
        # Show model in use.
        result2 = subprocess.run(
            ["ps", "-p", pids[0], "-o", "args="],
            capture_output=True,
            text=True,
        )
        if "--model" in result2.stdout:
            import re
            m = re.search(r"--model\s+(\S+)", result2.stdout)
            if m:
                print(f"   Model: {m.group(1)}")
    else:
        print("⚫ Not running.")


# ── helpers ────────────────────────────────────────────────────────────


def _find_and_kill_stale() -> int:
    """Find and kill any stale minty-box / hermes-listener processes.

    Returns:
        Number of processes killed.
    """
    killed = 0
    for pattern in STALE_PATTERNS:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            continue
        pids = result.stdout.strip().split()
        for pid_str in pids:
            try:
                pid = int(pid_str)
                if pid == os.getpid():
                    continue
                # Only kill our own processes.
                try:
                    pid_uid = os.stat(f"/proc/{pid}").st_uid
                    if pid_uid != os.getuid():
                        continue
                except (FileNotFoundError, PermissionError):
                    continue
                # Get any children.
                child_result = subprocess.run(
                    ["pgrep", "-P", str(pid)],
                    capture_output=True,
                    text=True,
                )
                children = (
                    child_result.stdout.strip().split()
                    if child_result.returncode == 0
                    else []
                )
                # Kill children first.
                for child_pid in children:
                    try:
                        os.kill(int(child_pid), signal.SIGKILL)
                        killed += 1
                    except (ProcessLookupError, PermissionError):
                        pass
                # Kill parent.
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                except (ProcessLookupError, PermissionError):
                    pass
            except (ValueError, ProcessLookupError):
                pass
    return killed


def _check_hardware() -> bool:
    """Verify the ReSpeaker Lite is present and ALSA sees it."""
    # lsusb check.
    result = subprocess.run(
        ["lsusb"],
        capture_output=True,
        text=True,
    )
    if "2886" not in result.stdout:
        return False

    # ALSA check.
    result2 = subprocess.run(
        ["cat", "/proc/asound/cards"],
        capture_output=True,
        text=True,
    )
    if "Lite" not in result2.stdout:
        return False

    return True


def _check_kokoro() -> bool:
    """Ping the Kokoro health endpoint."""
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "3", KOKORO_URL],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and '"status":"ok"' in result.stdout
    except Exception:
        return False


def _start_listener(
    model: str,
    threshold: float,
    warm: bool,
    captures: bool,
    no_tts: bool = False,
) -> None:
    """Launch main.py as a background subprocess.

    Uses ``subprocess.Popen`` so the CLI exits and leaves the
    listener running.  stdout/stderr are piped to /dev/null —
    captures go to ``captures/`` on disk.
    """
    cmd = [
        sys.executable,
        "-m", "minty_box.main",  # run as module (avoids path issues)
        "--model", model,
        "--threshold", str(threshold),
    ]
    if warm:
        cmd.append("--warm")
    if not captures:
        cmd.append("--no-captures")
    if no_tts:
        cmd.append("--no-tts")

    # Resolve model path relative to project root.
    import minty_box
    project_root = str(
        __import__("pathlib").Path(minty_box.__file__).parent.parent
    )

    subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # detach from parent process group
    )
    # Wait briefly for the subprocess to initialise and grab the audio device.
    time.sleep(4)
