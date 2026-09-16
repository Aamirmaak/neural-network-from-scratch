"""CLI tests for neuralearn command-line interface."""

from __future__ import annotations

import subprocess
import sys

import pytest

from neuralearn.cli import build_parser, main


class TestCLIParser:
    """Test the argparse parser construction."""

    def test_help_exits_cleanly(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_no_args_returns_zero(self):
        code = main([])
        assert code == 0

    def test_version_flag(self, capsys):
        code = main(["--version"])
        assert code == 0
        captured = capsys.readouterr()
        parts = captured.out.strip().split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_info_command(self, capsys):
        code = main(["info"])
        assert code == 0
        captured = capsys.readouterr()
        assert "neuralearn" in captured.out

    def test_example_command(self, capsys):
        code = main(["example"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Final loss" in captured.out
        assert "Accuracy" in captured.out

    def test_unknown_command_exits_with_error(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent"])
        assert exc_info.value.code == 2

    def test_invalid_option_exits_with_error(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--invalid-flag"])
        assert exc_info.value.code == 2


class TestCLISubprocess:
    """Test the CLI as an installed subprocess (requires package to be importable)."""

    def test_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "neuralearn.cli", "--version"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        parts = result.stdout.strip().split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "neuralearn.cli", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "--version" in result.stdout

    def test_info(self):
        result = subprocess.run(
            [sys.executable, "-m", "neuralearn.cli", "info"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "neuralearn" in result.stdout

    def test_example(self):
        result = subprocess.run(
            [sys.executable, "-m", "neuralearn.cli", "example"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "Final loss" in result.stdout

    def test_unknown_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "neuralearn.cli", "badcmd"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 2
        assert "error" in result.stderr.lower()
