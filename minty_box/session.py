"""Persistent Hermes session managed via tmux.

Keeps a single interactive ``hermes`` process alive in a detached tmux
session so that successive voice queries avoid cold-start latency
(Python spawn, Hermes boot, model load, tool init).
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time

logger = logging.getLogger(__name__)

SESSION_NAME = "minty-hermes"
PROMPT_CHAR = "▸"  # Hermes TUI prompt — signals response complete
CAPTURE_LINES = 200
POLL_INTERVAL = 2  # seconds


class WarmHermesSession:
    """Persistent ``hermes`` process in a detached tmux session.

    Start once with :meth:`start`, then call :meth:`query` for each
    voice utterance.  The session survives across queries — tools,
    model connection, and skills stay loaded.

    On ``start``, switches to a configured fast model (e.g. gemma4)
    so voice responses are quick, then warms the session with a
    trivial query.  The main Hermes config (cron, CLI, Discord) is
    not affected — this is a separate, voice-only instance.

    If the session dies or becomes unresponsive, :meth:`query` falls
    back to spawning a fresh ``hermes chat -q`` subprocess (with a
    generous 90s timeout).
    """

    # Model to switch to inside the voice session.
    _VOICE_MODEL = "gemma4:31b-cloud"

    def __init__(self) -> None:
        self._fallback_timeout = 90

    # ── public API ───────────────────────────────────────────────────

    def start(self, voice_model: str | None = None) -> bool:
        """Ensure the warm session is running, starting it if necessary.

        Creates a detached tmux session, boots Hermes, switches to the
        voice model, and warms the session with a trivial query.

        Args:
            voice_model: Model to switch to (default: ``_VOICE_MODEL``).

        Returns:
            ``True`` if a new session was created, ``False`` if one
            already existed.
        """
        if self._session_exists():
            logger.info(
                "Warm Hermes session '%s' already running.", SESSION_NAME
            )
            return False

        model = voice_model or self._VOICE_MODEL

        logger.info(
            "Starting warm Hermes session '%s' (voice-model=%s)...",
            SESSION_NAME, model,
        )

        # 1. Create detached tmux session.
        subprocess.run(
            [
                "tmux", "new-session", "-d", "-s", SESSION_NAME,
                "-x", "200", "-y", "40",
            ],
            check=True, capture_output=True, text=True,
        )

        # 2. Boot Hermes inside it.
        subprocess.run(
            ["tmux", "send-keys", "-t", SESSION_NAME, "hermes", "Enter"],
            check=True, capture_output=True, text=True,
        )
        time.sleep(6)  # Hermes boot: config, skills, model connect

        # 3. Switch to the fast voice model.
        logger.info("Switching voice session to %s...", model)
        # Send tokens separately to avoid tmux send-keys interpretation issues.
        subprocess.run(
            ["tmux", "send-keys", "-t", SESSION_NAME, "/model", "Space", model, "Enter"],
            check=True, capture_output=True, text=True,
        )
        time.sleep(6.0)

        # 4. Fire a warm-up query (fire-and-forget — first user query
        #    picks up the warm session naturally).
        logger.info("Priming voice session...")
        self._send_raw("Hello", wait=10.0)
        logger.info("Voice session ready.")

        return True

    def query(self, text: str) -> str:
        """Send *text* to the warm Hermes and return the response.

        Blocks until Hermes prints a new prompt or the fallback timeout
        is reached.

        Args:
            text: The transcribed utterance to send.

        Returns:
            Hermes's final text response.
        """
        if not text.strip():
            return "I didn't catch that."

        if not self._session_exists():
            logger.warning("Warm session gone — using cold fallback.")
            return self._cold_fallback(text)

        try:
            return self._warm_query(text)
        except Exception:
            logger.exception("Warm query failed — falling back to cold.")
            return self._cold_fallback(text)

    def stop(self) -> None:
        """Kill the tmux session, if it exists."""
        if self._session_exists():
            subprocess.run(
                ["tmux", "kill-session", "-t", SESSION_NAME],
                capture_output=True,
                text=True,
            )
            logger.info("Warm Hermes session stopped.")

    def is_running(self) -> bool:
        """Check whether the tmux session is alive."""
        return self._session_exists()

    # ── internals ────────────────────────────────────────────────────

    @staticmethod
    def _session_exists() -> bool:
        result = subprocess.run(
            ["tmux", "has-session", "-t", SESSION_NAME],
            capture_output=True,
        )
        return result.returncode == 0

    def _send_raw(self, text: str, wait: float = 2.0) -> None:
        """Send a command to the tmux session without waiting for prompt.

        Used for model switches and other commands that may not produce
        output detectable by ``_warm_query``'s prompt logic.

        Args:
            text: The command to send (Enter is appended automatically).
            wait: Seconds to sleep after sending.
        """
        subprocess.run(
            ["tmux", "send-keys", "-t", SESSION_NAME, text, "Enter"],
            check=True, capture_output=True, text=True,
        )
        time.sleep(wait)

    def _send_command(self, text: str) -> None:
        """Send a command and wait for prompt confirmation.

        Used for warm-up — not for user utterances (use :meth:`query`
        for those).
        """
        self._warm_query(text)

    def _warm_query(self, text: str) -> str:
        """Send text to interactive hermes, wait for prompt, extract response."""
        # Clear scrollback so we start fresh.
        subprocess.run(
            ["tmux", "clear-history", "-t", SESSION_NAME],
            capture_output=True,
            text=True,
        )

        # Send the query.
        subprocess.run(
            [
                "tmux", "send-keys", "-t", SESSION_NAME,
                text, "Enter",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Poll until we see the prompt reappear after our text.
        deadline = time.monotonic() + self._fallback_timeout
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL)
            pane = subprocess.run(
                [
                    "tmux", "capture-pane", "-t", SESSION_NAME, "-p",
                    "-S", f"-{CAPTURE_LINES}",
                ],
                capture_output=True,
                text=True,
            ).stdout

            response = _extract_response(pane, text)
            if response is not None:
                logger.info(
                    "Warm Hermes response (%d chars) for %r",
                    len(response), text[:60],
                )
                return response

        raise TimeoutError(
            f"Warm Hermes did not produce a prompt within "
            f"{self._fallback_timeout}s"
        )

    def _cold_fallback(self, text: str) -> str:
        """Spawn a fresh ``hermes chat -q`` subprocess."""
        try:
            result = subprocess.run(
                ["hermes", "chat", "-q", text, "-Q"],
                capture_output=True,
                text=True,
                timeout=self._fallback_timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error("Cold fallback timed out (%ds)", self._fallback_timeout)
            return "Sorry, that took too long. Could you ask again more simply?"
        except FileNotFoundError:
            logger.error("hermes binary not found for cold fallback")
            return "Sorry, my brain isn't connected right now."
        except Exception:
            logger.exception("Cold fallback failed")
            return "Sorry, something went wrong. Try again?"

        if result.returncode != 0:
            logger.error("Cold fallback exited %d", result.returncode)
            return "Sorry, I couldn't process that."

        response = result.stdout.strip()
        if not response:
            return "Hmm, I didn't get an answer to that."
        return response


def _extract_response(pane_text: str, query_text: str) -> str | None:
    """Try to extract Hermes's response from the pane capture.

    The response is the text between the echoed query line and the
    next prompt.  Returns ``None`` if the prompt hasn't reappeared
    yet (still processing).

    Args:
        pane_text: Full captured pane content.
        query_text: The query that was sent (to locate it in output).

    Returns:
        Response string, or ``None`` if not complete yet.
    """
    lines = pane_text.split("\n")

    # Find the query line (hermes echoes input after the prompt).
    query_idx: int | None = None
    query_clean = query_text.strip()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == query_clean:
            query_idx = i
            break
        # Hermes may prepend the prompt char + space.
        if stripped.endswith(query_clean) and PROMPT_CHAR in line:
            query_idx = i
            break

    if query_idx is None:
        # Query hasn't appeared yet (still in transit).
        return None

    # Look for the next prompt after the query line.
    prompt_idx: int | None = None
    for i in range(query_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped == PROMPT_CHAR or stripped.startswith(PROMPT_CHAR + " "):
            prompt_idx = i
            break

    if prompt_idx is None:
        return None  # still processing

    # Extract response lines between query and prompt.
    response_lines = lines[query_idx + 1 : prompt_idx]
    # Remove tool-call progress lines and empty lines.
    cleaned = [
        l for l in response_lines
        if l.strip() and not _is_tool_line(l)
    ]
    response = "\n".join(cleaned).strip()

    if not response:
        return "Hmm, I didn't get an answer to that."

    return response


def _is_tool_line(line: str) -> bool:
    """Heuristic: is this a tool-call progress/spinner line?"""
    stripped = line.strip()
    # Hermes tool progress indicators
    tool_signals = [
        "⏳", "🔧", "Tool:", "Running", "✓", "Executing",
        "Thinking", "Reasoning", "···", "…", "⠋", "⠙", "⠹", "⠸",
        "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
    ]
    return any(s in stripped for s in tool_signals)
