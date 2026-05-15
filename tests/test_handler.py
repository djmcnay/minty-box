"""Tests for minty_box.handler — DirectLLMHandler, _clean_for_speech, and legacy handlers."""

from __future__ import annotations

import json
import subprocess
from unittest import mock

import pytest

from minty_box.handler import (
    DirectLLMHandler,
    SingleShotHandler,
    WarmHermesHandler,
    _clean_for_speech,
    _get_system_prompt,
)


# ── _clean_for_speech ──────────────────────────────────────────────────


class TestCleanForSpeech:
    def test_passes_plain_text(self):
        assert _clean_for_speech("It is quarter to four.") == (
            "It is quarter to four."
        )

    def test_strips_bold_markdown(self):
        assert _clean_for_speech("It is **quarter** to four.") == (
            "It is quarter to four."
        )

    def test_strips_italic_markdown(self):
        assert _clean_for_speech("It is *quarter* to four.") == (
            "It is quarter to four."
        )

    def test_strips_triple_asterisk(self):
        assert _clean_for_speech("It is ***quarter*** to four.") == (
            "It is quarter to four."
        )

    def test_strips_markdown_link(self):
        assert _clean_for_speech(
            "Check [the docs](https://example.com) for info."
        ) == "Check the docs for info."

    def test_strips_inline_code(self):
        assert _clean_for_speech("Run `pip install` now.") == (
            "Run pip install now."
        )

    def test_strips_bullet_markers(self):
        # Newlines collapsed to spaces — correct for speech synthesis.
        assert _clean_for_speech("- Get milk\n- Get bread") == (
            "Get milk Get bread"
        )

    def test_strips_numbered_list(self):
        assert _clean_for_speech("1. First\n2. Second") == (
            "First Second"
        )

    def test_collapses_whitespace(self):
        assert _clean_for_speech("Hello    world\n\n\n  again") == (
            "Hello world again"
        )

    def test_empty_string(self):
        assert _clean_for_speech("") == ""

    def test_underscore_bold(self):
        assert _clean_for_speech("__important__ word") == "important word"


# ── _get_system_prompt ────────────────────────────────────────────────


class TestGetSystemPrompt:
    def test_loads_prompt_file(self, monkeypatch, tmp_path):
        prompt_file = tmp_path / "araminta_voice.txt"
        prompt_file.write_text("You are a test assistant.")
        monkeypatch.setattr(
            "minty_box.handler._PROMPT_PATH", prompt_file
        )
        monkeypatch.setattr("minty_box.handler._SYSTEM_PROMPT", None)
        result = _get_system_prompt()
        assert result == "You are a test assistant."

    def test_fallback_when_file_missing(self, monkeypatch, tmp_path):
        missing = tmp_path / "nonexistent.txt"
        monkeypatch.setattr(
            "minty_box.handler._PROMPT_PATH", missing
        )
        monkeypatch.setattr("minty_box.handler._SYSTEM_PROMPT", None)
        result = _get_system_prompt()
        assert "Minty" in result
        assert "British" in result


# ── DirectLLMHandler ──────────────────────────────────────────────────


class MockHTTPResponse:
    """Minimal urllib-style response for testing."""

    def __init__(self, body: str, status: int = 200):
        self._body = body.encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_llm_response(content: str) -> MockHTTPResponse:
    """Build a mock response with valid chat-completion JSON."""
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    return MockHTTPResponse(body)


class TestDirectLLMHandler:
    def test_process_returns_response(self):
        handler = DirectLLMHandler(model="test-model")
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_make_llm_response("It is quarter to four."),
        ):
            result = handler.process("What time is it?")
        assert result == "It is quarter to four."

    def test_process_empty_text(self):
        handler = DirectLLMHandler()
        result = handler.process("")
        assert result == "I didn't catch that."

    def test_process_whitespace_only(self):
        handler = DirectLLMHandler()
        result = handler.process("   ")
        assert result == "I didn't catch that."

    def test_process_connection_error(self):
        handler = DirectLLMHandler(timeout=5)
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=OSError("Connection refused"),
        ):
            result = handler.process("hello")
        assert "couldn't reach" in result.lower()

    def test_process_garbled_json(self):
        handler = DirectLLMHandler()
        with mock.patch(
            "urllib.request.urlopen",
            return_value=MockHTTPResponse("not json at all"),
        ):
            result = handler.process("hello")
        assert "garbled" in result.lower()

    def test_process_missing_choices_key(self):
        handler = DirectLLMHandler()
        with mock.patch(
            "urllib.request.urlopen",
            return_value=MockHTTPResponse(
                json.dumps({"not_choices": []})
            ),
        ):
            result = handler.process("hello")
        assert "garbled" in result.lower()

    def test_process_cleans_markdown(self):
        """LLM response with markdown should be cleaned for speech."""
        handler = DirectLLMHandler()
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_make_llm_response(
                "It is **quarter** to four.  See [docs](http://x.com)."
            ),
        ):
            result = handler.process("What time is it?")
        assert "**" not in result
        assert "[docs]" not in result
        assert "quarter to four" in result

    def test_custom_model_and_tokens(self):
        handler = DirectLLMHandler(
            model="custom-model:latest",
            max_tokens=200,
            temperature=0.3,
            timeout=45,
        )
        assert handler._model == "custom-model:latest"
        assert handler._max_tokens == 200
        assert handler._temperature == 0.3
        assert handler._timeout == 45


# ── Legacy handlers (unchanged behaviour) ─────────────────────────────


def _make_completed(returncode=0, stdout="Hello, David.", stderr=""):
    return subprocess.CompletedProcess(
        args=["hermes", "chat", "-q", "test", "-Q"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class TestSingleShotHandler:
    def test_successful_response(self):
        handler = SingleShotHandler()
        with mock.patch("subprocess.run", return_value=_make_completed()):
            result = handler.process("What time is it?")
        assert result == "Hello, David."

    def test_default_timeout_is_90(self):
        handler = SingleShotHandler()
        assert handler._timeout == 90

    def test_file_not_found(self):
        handler = SingleShotHandler()
        with mock.patch(
            "subprocess.run",
            side_effect=FileNotFoundError("hermes"),
        ):
            result = handler.process("hello")
        assert "brain" in result.lower()


class TestWarmHermesHandler:
    def test_delegates_to_session_query(self):
        mock_session = mock.MagicMock()
        mock_session.query.return_value = "It is 3:45 PM."
        handler = WarmHermesHandler(mock_session)
        result = handler.process("What time is it?")
        mock_session.query.assert_called_once_with("What time is it?")
        assert result == "It is 3:45 PM."

    def test_empty_text(self):
        mock_session = mock.MagicMock()
        handler = WarmHermesHandler(mock_session)
        result = handler.process("")
        assert result == "I didn't catch that."
        mock_session.query.assert_not_called()

    def test_session_exception_returns_fallback(self):
        mock_session = mock.MagicMock()
        mock_session.query.side_effect = RuntimeError("tmux died")
        handler = WarmHermesHandler(mock_session)
        result = handler.process("hello")
        assert "something went wrong" in result.lower()
