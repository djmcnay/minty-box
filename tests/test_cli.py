"""Tests for minty_box.cli — CLI launcher."""

from __future__ import annotations

from unittest import mock

import pytest

from minty_box.cli import main
from minty_box.cli import _check_hardware
from minty_box.cli import _check_kokoro
from minty_box.cli import _find_and_kill_stale
from minty_box.cli import _start_listener


# ── _find_and_kill_stale ───────────────────────────────────────────────


class TestFindAndKillStale:
    def test_no_stale_processes(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1  # pgrep finds nothing
            killed = _find_and_kill_stale()
        assert killed == 0

    @mock.patch("builtins.print")
    def test_kills_stale_main_py(self, _mock_print):
        """pgrep returns PIDs, os.kill cleans them up."""
        with mock.patch(
            "subprocess.run",
            side_effect=[
                # pgrep -f main.py → finds two PIDs
                mock.MagicMock(returncode=0, stdout="12345\n12346\n"),
                # pgrep -P 12345 → children of first
                mock.MagicMock(returncode=0, stdout="12347\n"),
                # pgrep -P 12346 → children of second
                mock.MagicMock(returncode=1, stdout=""),
                # pgrep for other patterns → nothing
                mock.MagicMock(returncode=1, stdout=""),
                mock.MagicMock(returncode=1, stdout=""),
            ],
        ), mock.patch("os.kill") as mock_kill, mock.patch(
            "os.getpid", return_value=99999
        ), mock.patch("os.getuid", return_value=1000), mock.patch(
            "os.stat"
        ) as mock_stat:
            # All /proc/ entries belong to our UID.
            mock_stat.return_value.st_uid = 1000
            killed = _find_and_kill_stale()
        # 12345 + child 12347 + 12346 = 3 kills
        assert killed == 3
        assert mock_kill.call_count == 3


# ── _check_hardware ────────────────────────────────────────────────────


class TestCheckHardware:
    @mock.patch("builtins.print")
    def test_all_ok(self, _mock_print):
        with mock.patch(
            "subprocess.run",
            side_effect=[
                mock.MagicMock(stdout="Bus 001 Device 049: ID 2886:0019 ReSpeaker Lite"),
                mock.MagicMock(stdout=" 2 [Lite           ]: USB-Audio - ReSpeaker Lite"),
            ],
        ):
            ok = _check_hardware()
        assert ok is True

    @mock.patch("builtins.print")
    def test_reSpeaker_missing(self, _mock_print):
        with mock.patch(
            "subprocess.run",
            side_effect=[
                mock.MagicMock(stdout="Bus 001 Device 001: Linux Foundation root hub"),
                mock.MagicMock(stdout=""),
            ],
        ):
            ok = _check_hardware()
        assert ok is False


# ── _check_kokoro ──────────────────────────────────────────────────────


class TestCheckKokoro:
    @mock.patch("builtins.print")
    def test_healthy(self, _mock_print):
        with mock.patch(
            "subprocess.run",
            return_value=mock.MagicMock(
                stdout='{"status":"ok","provider":"kokoro-onnx"}',
                returncode=0,
            ),
        ):
            ok = _check_kokoro()
        assert ok is True

    @mock.patch("builtins.print")
    def test_unhealthy(self, _mock_print):
        with mock.patch(
            "subprocess.run",
            side_effect=Exception("Connection refused"),
        ):
            ok = _check_kokoro()
        assert ok is False


# ── _start_listener ────────────────────────────────────────────────────


class TestStartListener:
    def test_starts_background_process(self):
        with mock.patch("subprocess.Popen") as mock_popen, \
             mock.patch("builtins.print"), \
             mock.patch("time.sleep"):
            _start_listener(
                model="models/Araminta.onnx",
                threshold=0.3,
                warm=True,
                captures=False,
            )
        mock_popen.assert_called_once()
        # Module-based invocation: python -m minty_box.main
        args = mock_popen.call_args[0][0]
        assert "-m" in args
        assert "minty_box.main" in args
        assert "--warm" in args
        assert "--model" in args


# ── main CLI ───────────────────────────────────────────────────────────


class TestCLIMain:
    def test_status_shows_process_info(self):
        with mock.patch("builtins.print") as mock_print, \
             mock.patch("sys.argv", ["minty-box", "status"]), \
             mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                returncode=0, stdout="12345\n"
            )
            main()
        assert mock_print.call_count > 0

    def test_status_no_process(self):
        with mock.patch("builtins.print"), \
             mock.patch("sys.argv", ["minty-box", "status"]), \
             mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                returncode=1, stdout=""
            )
            main()

    @mock.patch("builtins.print")
    @mock.patch("minty_box.session.WarmHermesSession")
    def test_start_runs_full_sequence(self, mock_session_cls, _mock_print):
        mock_session = mock.MagicMock()
        mock_session.start.return_value = True
        mock_session_cls.return_value = mock_session

        with mock.patch("sys.argv", [
            "minty-box", "start",
            "--model", "models/Araminta.onnx",
            "--threshold", "0.3",
            "--warm",
        ]), \
             mock.patch(
                 "minty_box.cli._find_and_kill_stale", return_value=0
             ), \
             mock.patch(
                 "minty_box.cli._check_hardware", return_value=True
             ), \
             mock.patch(
                 "minty_box.cli._check_kokoro", return_value=True
             ), \
             mock.patch("minty_box.cli._start_listener"), \
             mock.patch("time.sleep"):
            main()
        mock_session.start.assert_called_once()

    @mock.patch("builtins.print")
    def test_stop_kills_and_verifies(self, _mock_print):
        with mock.patch("sys.argv", ["minty-box", "stop"]), \
             mock.patch(
                 "minty_box.cli._find_and_kill_stale", return_value=2
             ), \
             mock.patch(
                 "minty_box.session.WarmHermesSession"
             ) as mock_session_cls:
            mock_session = mock.MagicMock()
            mock_session_cls.return_value = mock_session
            main()
        mock_session.stop.assert_called_once()
