"""Tests for minty_box.session — WarmHermesSession and helpers."""

from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from minty_box.session import (
    WarmHermesSession,
    _extract_response,
    _is_tool_line,
)


# ── _is_tool_line ──────────────────────────────────────────────────────


class TestIsToolLine:
    def test_spinner_line(self):
        assert _is_tool_line("⠋ Thinking...") is True

    def test_tool_call_line(self):
        assert _is_tool_line("  🔧 web_search(query='hello')") is True

    def test_tool_result_line(self):
        assert _is_tool_line("      ✓ Done") is True

    def test_normal_text_is_not_tool(self):
        assert _is_tool_line("It is 3:45 PM.") is False

    def test_empty_line(self):
        assert _is_tool_line("   ") is False  # stripped is empty


# ── _extract_response ──────────────────────────────────────────────────


class TestExtractResponse:
    def test_extracts_simple_response(self):
        pane = (
            "▸ some earlier output\n"
            "What time is it?\n"
            "It is quarter to four.\n"
            "▸ "
        )
        result = _extract_response(pane, "What time is it?")
        assert result == "It is quarter to four."

    def test_prompt_not_yet_present_returns_none(self):
        pane = (
            "▸ old stuff\n"
            "What time is it?\n"
            "⠋ Thinking..."
        )
        result = _extract_response(pane, "What time is it?")
        assert result is None

    def test_strips_tool_lines_from_response(self):
        pane = (
            "▸ \n"
            "What time is it?\n"
            "⠋ Thinking...\n"
            "🔧 web_search(time)\n"
            "It is 3:45 PM.\n"
            "▸ "
        )
        result = _extract_response(pane, "What time is it?")
        assert result == "It is 3:45 PM."

    def test_query_not_found_returns_none(self):
        pane = "▸ \nSomething else entirely\n▸ "
        result = _extract_response(pane, "What time is it?")
        assert result is None

    def test_empty_response_falls_back(self):
        pane = (
            "▸ \nWhat time is it?\n▸ "
        )
        result = _extract_response(pane, "What time is it?")
        assert "didn't get an answer" in result.lower()


# ── WarmHermesSession (unit tests, no real tmux) ──────────────────────


class TestWarmHermesSession:
    def test_session_exists_true(self):
        session = WarmHermesSession()
        with mock.patch.object(
            session, "_session_exists", return_value=True
        ):
            assert session.is_running() is True

    def test_session_exists_false(self):
        session = WarmHermesSession()
        with mock.patch.object(
            session, "_session_exists", return_value=False
        ):
            assert session.is_running() is False

    def test_start_when_already_running(self):
        session = WarmHermesSession()
        with mock.patch.object(
            session, "_session_exists", return_value=True
        ), mock.patch("subprocess.run") as mock_run:
            created = session.start()
        assert created is False
        mock_run.assert_not_called()

    def test_start_creates_new_session(self):
        session = WarmHermesSession()
        with mock.patch.object(
            session, "_session_exists", return_value=False
        ), mock.patch("subprocess.run") as mock_run, mock.patch(
            "time.sleep"
        ):
            created = session.start()
        assert created is True
        assert mock_run.call_count >= 2  # new-session + send-keys

    def test_stop_kills_session(self):
        session = WarmHermesSession()
        with mock.patch.object(
            session, "_session_exists", return_value=True
        ), mock.patch("subprocess.run") as mock_run:
            session.stop()
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "kill-session" in args

    def test_query_empty_text(self):
        session = WarmHermesSession()
        result = session.query("")
        assert result == "I didn't catch that."

    def test_query_whitespace_text(self):
        session = WarmHermesSession()
        result = session.query("   ")
        assert result == "I didn't catch that."

    def test_query_falls_back_when_session_gone(self):
        session = WarmHermesSession()
        with mock.patch.object(
            session, "_session_exists", return_value=False
        ), mock.patch.object(session, "_cold_fallback") as mock_cold:
            mock_cold.return_value = "It is 3 PM."
            result = session.query("What time is it?")
        mock_cold.assert_called_once_with("What time is it?")
        assert result == "It is 3 PM."

    def test_cold_fallback_success(self):
        session = WarmHermesSession()
        completed = subprocess.CompletedProcess(
            args=["hermes", "chat", "-q", "test", "-Q"],
            returncode=0,
            stdout="The answer is 42.",
            stderr="",
        )
        with mock.patch("subprocess.run", return_value=completed):
            result = session._cold_fallback("What is the answer?")
        assert result == "The answer is 42."

    def test_cold_fallback_timeout(self):
        session = WarmHermesSession()
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("hermes", 90),
        ):
            result = session._cold_fallback("long query")
        assert "too long" in result.lower()

    def test_cold_fallback_file_not_found(self):
        session = WarmHermesSession()
        with mock.patch(
            "subprocess.run",
            side_effect=FileNotFoundError("hermes"),
        ):
            result = session._cold_fallback("hello")
        assert "brain" in result.lower()

    def test_cold_fallback_nonzero_exit(self):
        session = WarmHermesSession()
        completed = subprocess.CompletedProcess(
            args=["hermes"],
            returncode=1,
            stdout="",
            stderr="Error",
        )
        with mock.patch("subprocess.run", return_value=completed):
            result = session._cold_fallback("hello")
        assert "couldn't process" in result.lower()

    def test_cold_fallback_empty_stdout(self):
        session = WarmHermesSession()
        completed = subprocess.CompletedProcess(
            args=["hermes", "chat", "-q", "test"],
            returncode=0,
            stdout="",
            stderr="",
        )
        with mock.patch("subprocess.run", return_value=completed):
            result = session._cold_fallback("hello")
        assert "didn't get an answer" in result.lower()
