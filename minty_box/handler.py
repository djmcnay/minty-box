"""Handlers for wake-word-triggered utterances.

Routes transcribed utterances to the Hermes agent backend (the real
Araminta, with SOUL.md, memory, and skills) via the gateway's built-in
API server."""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

DEFAULT_LLM_TIMEOUT = 30  # agent may need 2-3 API calls + tool turns
MAX_TOKENS = 80  # ~3 spoken sentences

# ── Hermes API server ──────────────────────────────────────────────────
HERMES_API = "http://127.0.0.1:8643/v1/chat/completions"


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

    The ``model`` parameter is accepted for API compatibility but is
    intentionally ignored: per-request model routing does not work with
    the Hermes agent loop, and the gateway default (``gemma4-voice``)
    is pre-configured for voice sessions.

    Target latency: <2s with fast models (gemma4), 3–8s with heavy
    models (deepseek), including 0–2 tool turns.
    """

    def __init__(
        self,
        model: str | None = None,  # noqa: ARG002  # ignored — gateway decides
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_tokens: int = MAX_TOKENS,
        warm: bool = True,
    ) -> None:
        self._timeout = timeout
        self._max_tokens = max_tokens
        if warm:
            self._warm_up()

    def _warm_up(self) -> None:
        """Send a trivial query to prime the agent session.

        Uses the Hermes default model.  Blocking — the caller accepts
        the startup delay.
        """
        logger.info("Warming Hermes agent session (this may take 10-30s)...")
        try:
            self._call_api("Hello")
            logger.info("Hermes agent session warm.")
        except Exception:
            logger.warning("Agent warm-up failed; first query will be slow")

    def _call_api(self, text: str) -> str:
        """POST to the Hermes API server.  Returns response text."""
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

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read().decode("utf-8")

        data = json.loads(body)
        return data["choices"][0]["message"]["content"]

    def process(self, text: str) -> str:
        if not text.strip():
            return "I didn't catch that."

        try:
            result = self._call_api(text)
        except Exception as e:
            logger.error("Hermes API call failed: %s", e)
            return "Sorry, my brain isn't connected right now. Try again?"

        response = _clean_for_speech(result)
        logger.info(
            "Hermes API response (%d chars) for %r",
            len(response), text[:60],
        )
        return response


# ── Response cleaning for TTS ──────────────────────────────────────────

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
