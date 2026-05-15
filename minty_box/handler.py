"""Handlers for wake-word-triggered utterances.

Routes transcribed utterances to the Hermes agent backend (the real
Araminta, with SOUL.md, memory, and skills) via the gateway's built-in
API server.  Legacy direct-LLM and subprocess handlers kept for fallback.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minty_box.session import WarmHermesSession

logger = logging.getLogger(__name__)

DEFAULT_HERMES_TIMEOUT = 90
DEFAULT_LLM_TIMEOUT = 30  # agent may need 2-3 API calls + tool turns
MAX_TOKENS = 80  # ~3 spoken sentences

# ── Hermes API server ──────────────────────────────────────────────────
HERMES_API = "http://127.0.0.1:8642/v1/chat/completions"


# ═══════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════


class BaseHandler(ABC):
    """Abstract handler for a wake-word-triggered utterance."""

    @abstractmethod
    def process(self, text: str) -> str:
        """Process transcribed text and return a response string."""
        ...


# ═══════════════════════════════════════════════════════════════════════
# Hermes API handler — the REAL agent (SOUL.md, memory, skills, tools)
# ═══════════════════════════════════════════════════════════════════════


class HermesAPIHandler(BaseHandler):
    """Routes utterances to the real Hermes agent via its API server.

    Hits ``/v1/chat/completions`` on the Hermes gateway's built-in
    API server — the *same agent* that answers on Discord and Telegram.
    No system prompt needed: Hermes already carries SOUL.md, memory,
    skills, and full tool access.

    Target latency: 3–8 seconds for typical voice queries (agent loop
    with 0–2 tool calls).
    """

    def __init__(
        self,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self._timeout = timeout
        self._max_tokens = max_tokens

    def process(self, text: str) -> str:
        if not text.strip():
            return "I didn't catch that."

        payload = json.dumps(
            {
                "model": "hermes-agent",
                "messages": [
                    {"role": "user", "content": text.strip()},
                ],
                "max_tokens": self._max_tokens,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            HERMES_API,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
        except Exception as e:
            logger.error("Hermes API call failed: %s", e)
            return "Sorry, my brain isn't connected right now. Try again?"

        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error("Failed to parse Hermes API response: %s", e)
            logger.debug("Raw response: %s", body[:500])
            return "Sorry, I got a garbled response. Try again?"

        response = _clean_for_speech(content)

        logger.info(
            "Hermes API response (%d chars) for %r",
            len(response),
            text[:60],
        )
        return response


# ═══════════════════════════════════════════════════════════════════════
# Legacy: direct LLM handler (generic model + system-prompt, NOT Araminta)
# ═══════════════════════════════════════════════════════════════════════

OLLAMA_API = "http://127.0.0.1:11434/v1/chat/completions"
VOICE_MODEL = "gemini-3-flash-preview:latest"
TEMPERATURE = 0.7

_PROMPT_PATH = Path(__file__).parent / "prompts" / "araminta_voice.txt"
_SYSTEM_PROMPT: str | None = None


def _get_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        try:
            _SYSTEM_PROMPT = _PROMPT_PATH.read_text().strip()
        except FileNotFoundError:
            logger.error("Voice prompt not found at %s", _PROMPT_PATH)
            _SYSTEM_PROMPT = (
                "You are Minty, a British AI assistant. "
                "Keep responses to 1-3 sentences. Be dry and warm."
            )
    return _SYSTEM_PROMPT


class DirectLLMHandler(BaseHandler):
    """Handles utterances by POSTing directly to a fast cloud LLM.

    No Hermes agent loop.  No tmux.  No pane-scraping.  A single
    HTTP POST to an OpenAI-compatible chat completions endpoint
    with Araminta's voice persona as the system prompt.

    Target latency: 2–5 seconds for the LLM call, ~10s end-to-end
    including STT, TTS, and playback.

    Uses Ollama's local OpenAI-compatible API gateway, which proxies
    to cloud models (Gemini Flash, DeepSeek Flash, etc.) depending
    on which model tag is requested.
    """

    def __init__(
        self,
        model: str = VOICE_MODEL,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        timeout: int = DEFAULT_LLM_TIMEOUT,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    def process(self, text: str) -> str:
        if not text.strip():
            return "I didn't catch that."

        system_prompt = _get_system_prompt()

        payload = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text.strip()},
                ],
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            OLLAMA_API,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            return "Sorry, I couldn't reach my brain. Try again?"

        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error("Failed to parse LLM response: %s", e)
            logger.debug("Raw response: %s", body[:500])
            return "Sorry, I got a garbled response. Try again?"

        response = _clean_for_speech(content)

        logger.info(
            "LLM response (%d chars, model=%s) for %r",
            len(response),
            self._model,
            text[:60],
        )
        return response


# ═══════════════════════════════════════════════════════════════════════
# Legacy: cold Hermes subprocess handler
# ═══════════════════════════════════════════════════════════════════════


class SingleShotHandler(BaseHandler):
    """Handles utterances by spawning a one-shot ``hermes chat -q``.

    Each invocation is a **fresh session** — stateless, no conversation
    history carried across calls.  Suitable for single-shot wake word
    mode (``"Hey Minty"`` / current ``"Araminta"`` mapping).
    """

    def __init__(self, timeout: int = DEFAULT_HERMES_TIMEOUT) -> None:
        self._timeout = timeout

    def process(self, text: str) -> str:
        if not text.strip():
            return "I didn't catch that."

        try:
            result = subprocess.run(
                ["hermes", "chat", "-q", text, "-Q"],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except FileNotFoundError:
            logger.error("hermes binary not found on PATH")
            return "Sorry, my brain isn't connected right now."
        except subprocess.TimeoutExpired:
            logger.error(
                "Hermes timed out after %ds — query was: %r",
                self._timeout,
                text,
            )
            return "Sorry, that took too long. Could you ask again more simply?"
        except Exception:
            logger.exception("Hermes subprocess failed")
            return "Sorry, something went wrong. Try again?"

        if result.returncode != 0:
            logger.error(
                "Hermes exited %d. stderr: %s",
                result.returncode,
                result.stderr[:200],
            )
            return "Sorry, I couldn't process that."

        response = result.stdout.strip()
        if not response:
            logger.warning("Hermes returned empty response for: %r", text)
            return "Hmm, I didn't get an answer to that."

        logger.info(
            "Hermes response (%d chars) for %r", len(response), text[:60]
        )
        return response


class WarmHermesHandler(BaseHandler):
    """Handles utterances via a persistent, warm Hermes tmux session."""

    def __init__(self, session: WarmHermesSession) -> None:
        self._session = session

    def process(self, text: str) -> str:
        if not text.strip():
            return "I didn't catch that."

        try:
            return self._session.query(text)
        except Exception:
            logger.exception("Warm handler failed — no fallback available")
            return "Sorry, something went wrong. Try again?"


# ═══════════════════════════════════════════════════════════════════════
# Response cleaning for TTS
# ═══════════════════════════════════════════════════════════════════════


def _clean_for_speech(text: str) -> str:
    """Strip markdown and formatting artefacts for TTS synthesis."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s*[-*•·]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text
