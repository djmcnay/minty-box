"""Tests for minty_box.handler — HermesAPIHandler and _clean_for_speech."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from minty_box.handler import (
    HermesAPIHandler,
    _clean_for_speech,
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


# ── HermesAPIHandler ───────────────────────────────────────────────────


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


def _make_api_response(content: str) -> MockHTTPResponse:
    """Build a mock Hermes API response with valid chat-completion JSON."""
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


class TestHermesAPIHandler:
    def test_process_returns_response(self):
        handler = HermesAPIHandler(warm=False)
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_make_api_response("It is quarter to four."),
        ):
            result = handler.process("What time is it?")
        assert result == "It is quarter to four."

    def test_process_empty_text(self):
        handler = HermesAPIHandler(warm=False)
        result = handler.process("")
        assert result == "I didn't catch that."

    def test_process_whitespace_only(self):
        handler = HermesAPIHandler(warm=False)
        result = handler.process("   ")
        assert result == "I didn't catch that."

    def test_process_connection_error(self):
        handler = HermesAPIHandler(warm=False, timeout=5)
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=OSError("Connection refused"),
        ):
            result = handler.process("hello")
        assert "brain isn't connected" in result.lower()

    def test_process_malformed_json(self):
        handler = HermesAPIHandler(warm=False)
        with mock.patch(
            "urllib.request.urlopen",
            return_value=MockHTTPResponse("not json at all"),
        ):
            result = handler.process("hello")
        assert "brain isn't connected" in result.lower()

    def test_process_missing_choices_key(self):
        handler = HermesAPIHandler(warm=False)
        with mock.patch(
            "urllib.request.urlopen",
            return_value=MockHTTPResponse(
                json.dumps({"not_choices": []})
            ),
        ):
            result = handler.process("hello")
        assert "brain isn't connected" in result.lower()

    def test_process_cleans_markdown(self):
        handler = HermesAPIHandler(warm=False)
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_make_api_response(
                "It is **quarter** to four.  See [docs](http://x.com)."
            ),
        ):
            result = handler.process("What time is it?")
        assert "**" not in result
        assert "[docs]" not in result
        assert "quarter to four" in result

    def test_model_parameter_ignored(self):
        handler = HermesAPIHandler(model="some-model:latest", warm=False)
        assert not hasattr(handler, "_voice_model")

    def test_warm_up_calls_api(self):
        handler = HermesAPIHandler(warm=False)
        with mock.patch.object(
            handler, "_call_api", return_value="Hello there."
        ) as mock_call:
            handler._warm_up()
        mock_call.assert_called_once_with("Hello")

    def test_call_api_payload_structure(self):
        handler = HermesAPIHandler(warm=False, max_tokens=120)
        captured = {}

        def capture(req, **kwargs):
            captured["body"] = req.data
            return _make_api_response("OK")

        with mock.patch("urllib.request.urlopen", side_effect=capture):
            handler._call_api("Test query")

        payload = json.loads(captured["body"])
        assert payload["model"] == "hermes-agent"
        assert payload["messages"] == [
            {"role": "user", "content": "Test query"}
        ]
        assert payload["max_tokens"] == 120
