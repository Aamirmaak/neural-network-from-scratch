"""Tests for the neuralearn demo CLI command."""

from __future__ import annotations

import subprocess
import sys

import pytest

from neuralearn.cli import build_parser, main, _build_xor_demo_model, _make_xor_data


class TestDemoCommand:
    """Test the demo CLI command."""

    def test_demo_succeeds(self, capsys):
        code = main(["demo", "--epochs", "50"])
        assert code == 0
        captured = capsys.readouterr()
        assert "NeuraLearn Demo" in captured.out
        assert "XOR" in captured.out

    def test_demo_shows_model_info(self, capsys):
        code = main(["demo", "--epochs", "10"])
        assert code == 0
        captured = capsys.readouterr()
        assert "MLP" in captured.out
        assert "Parameters" in captured.out

    def test_demo_shows_training_progress(self, capsys):
        code = main(["demo", "--epochs", "20"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Epoch" in captured.out
        assert "loss" in captured.out

    def test_demo_shows_predictions(self, capsys):
        code = main(["demo", "--epochs", "10"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Predictions" in captured.out
        assert "Correct" in captured.out

    def test_demo_shows_final_result(self, capsys):
        code = main(["demo", "--epochs", "300"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Result" in captured.out
        assert "Accuracy" in captured.out

    def test_demo_deterministic(self):
        """Running demo twice produces same final loss."""
        import io
        from contextlib import redirect_stdout

        f1 = io.StringIO()
        with redirect_stdout(f1):
            main(["demo", "--epochs", "50"])
        out1 = f1.getvalue()

        f2 = io.StringIO()
        with redirect_stdout(f2):
            main(["demo", "--epochs", "50"])
        out2 = f2.getvalue()

        assert out1 == out2

    def test_demo_custom_hidden(self, capsys):
        code = main(["demo", "--epochs", "10", "--hidden", "4"])
        assert code == 0
        captured = capsys.readouterr()
        assert "2 -> 4 -> 1" in captured.out

    def test_demo_custom_lr(self, capsys):
        code = main(["demo", "--epochs", "10", "--lr", "0.1"])
        assert code == 0
        captured = capsys.readouterr()
        assert "lr=0.1" in captured.out

    def test_demo_plot_without_matplotlib(self, capsys, monkeypatch):
        """Plot flag with missing matplotlib should not crash."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        code = main(["demo", "--epochs", "10", "--plot"])
        assert code == 0
        captured = capsys.readouterr()
        assert "matplotlib not installed" in captured.out

    def test_demo_plot_with_matplotlib(self, capsys, tmp_path):
        """Plot flag with matplotlib should generate files."""
        code = main(["demo", "--epochs", "50", "--plot", "--output", str(tmp_path)])
        assert code == 0
        assert (tmp_path / "demo_loss_curve.png").exists()
        assert (tmp_path / "demo_predictions.png").exists()


class TestDemoHelpers:
    """Test internal demo helper functions."""

    def test_build_model(self):
        model = _build_xor_demo_model(hidden_size=8)
        assert len(model.parameters()) == 33  # 2*8+8 + 8*1+1

    def test_build_model_custom_hidden(self):
        model = _build_xor_demo_model(hidden_size=4)
        # Linear(2,4): 2*4+4=12, Linear(4,1): 4*1+1=5, total=17
        assert len(model.parameters()) == 17

    def test_make_xor_data(self):
        inputs, targets = _make_xor_data()
        assert len(inputs) == 4
        assert len(targets) == 4
        assert len(inputs[0]) == 2
        assert len(targets[0]) == 1


class TestDemoSubprocess:
    """Test demo via subprocess (installed package)."""

    def test_demo_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "neuralearn.cli", "demo", "--epochs", "20"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "NeuraLearn Demo" in result.stdout
        assert "Result" in result.stdout
