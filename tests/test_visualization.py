"""
Tests for visualization utilities.

Covers:
    1. Plot functions accept valid histories
    2. Files are created successfully
    3. Invalid inputs produce clear errors
    4. Empty histories are handled
    5. Output directories are created
    6. Plotting does not mutate inputs
"""

import os
import tempfile

import pytest

from neuralearn.visualization import (
    plot_loss,
    plot_train_test_curve,
    plot_regression_fit,
    plot_xor_predictions,
    plot_comparison,
    plot_accuracy,
)


@pytest.fixture
def tmp_plot_dir():
    """Create a temporary directory for plot output."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def xor_history():
    return {
        "epoch": [1, 100, 200, 300, 400],
        "loss": [0.72, 0.69, 0.69, 0.61, 0.28],
        "accuracy": [50.0, 50.0, 75.0, 75.0, 100.0],
    }


@pytest.fixture
def regression_history():
    return {
        "epoch": [1, 50, 100, 150, 200],
        "train_mse": [0.71, 0.025, 0.023, 0.12, 0.019],
        "test_mse": [0.30, 0.027, 0.032, 0.090, 0.041],
    }


# ---------------------------------------------------------------------------
# 1. plot_loss
# ---------------------------------------------------------------------------

class TestPlotLoss:
    def test_creates_file(self, xor_history, tmp_plot_dir):
        path = plot_loss(xor_history, save_path=os.path.join(tmp_plot_dir, "loss.png"))
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_default_save_path(self, xor_history):
        path = plot_loss(xor_history)
        assert os.path.exists(path)
        os.remove(path)

    def test_custom_loss_key(self, tmp_plot_dir):
        history = {"epoch": [1, 2], "train_mse": [0.5, 0.1]}
        path = plot_loss(history, loss_key="train_mse",
                         save_path=os.path.join(tmp_plot_dir, "mse.png"))
        assert os.path.exists(path)

    def test_empty_history_raises(self):
        with pytest.raises(ValueError, match="no data"):
            plot_loss({"loss": []})

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="no data"):
            plot_loss({"epoch": [1, 2]})

    def test_no_epoch_uses_index(self, tmp_plot_dir):
        history = {"loss": [0.5, 0.3, 0.1]}
        path = plot_loss(history, save_path=os.path.join(tmp_plot_dir, "idx.png"))
        assert os.path.exists(path)

    def test_does_not_mutate_history(self, xor_history):
        original = {k: list(v) for k, v in xor_history.items()}
        plot_loss(xor_history)
        assert xor_history == original


# ---------------------------------------------------------------------------
# 2. plot_train_test_curve
# ---------------------------------------------------------------------------

class TestPlotTrainTestCurve:
    def test_creates_file(self, regression_history, tmp_plot_dir):
        path = plot_train_test_curve(
            regression_history,
            save_path=os.path.join(tmp_plot_dir, "tt.png"),
        )
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_missing_train_key_raises(self):
        with pytest.raises(ValueError, match="no data"):
            plot_train_test_curve({"test_mse": [0.1]})

    def test_missing_test_key_raises(self):
        with pytest.raises(ValueError, match="no data"):
            plot_train_test_curve({"train_mse": [0.1]})


# ---------------------------------------------------------------------------
# 3. plot_regression_fit
# ---------------------------------------------------------------------------

class TestPlotRegressionFit:
    def test_creates_file(self, tmp_plot_dir):
        x_train = [0.0, 1.0, 2.0]
        y_train = [0.0, 1.0, 4.0]
        x_test = [0.5, 1.5]
        y_test = [0.25, 2.25]
        preds = [0.3, 2.1]
        path = plot_regression_fit(
            x_train, y_train, x_test, y_test, preds,
            save_path=os.path.join(tmp_plot_dir, "reg.png"),
        )
        assert os.path.exists(path)

    def test_with_target_fn(self, tmp_plot_dir):
        import math
        path = plot_regression_fit(
            [0.0, 1.0], [0.0, 1.0],
            [0.5], [0.5],
            [0.48],
            target_fn=math.sin,
            save_path=os.path.join(tmp_plot_dir, "reg_fn.png"),
        )
        assert os.path.exists(path)

    def test_empty_data_creates_file(self, tmp_plot_dir):
        path = plot_regression_fit(
            [], [], [], [], [],
            save_path=os.path.join(tmp_plot_dir, "empty.png"),
        )
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# 4. plot_xor_predictions
# ---------------------------------------------------------------------------

class TestPlotXorPredictions:
    def test_creates_file(self, tmp_plot_dir):
        results = [
            {"input": [0.0, 0.0], "target": 0.0, "prediction": 0.003,
             "predicted_class": 0.0, "correct": True},
            {"input": [0.0, 1.0], "target": 1.0, "prediction": 0.995,
             "predicted_class": 1.0, "correct": True},
            {"input": [1.0, 0.0], "target": 1.0, "prediction": 0.996,
             "predicted_class": 1.0, "correct": True},
            {"input": [1.0, 1.0], "target": 0.0, "prediction": 0.005,
             "predicted_class": 0.0, "correct": True},
        ]
        path = plot_xor_predictions(
            results,
            save_path=os.path.join(tmp_plot_dir, "xor.png"),
        )
        assert os.path.exists(path)

    def test_incorrect_predictions(self, tmp_plot_dir):
        results = [
            {"input": [0.0, 0.0], "target": 0.0, "prediction": 0.6,
             "predicted_class": 1.0, "correct": False},
        ]
        path = plot_xor_predictions(
            results,
            save_path=os.path.join(tmp_plot_dir, "xor_wrong.png"),
        )
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# 5. plot_comparison
# ---------------------------------------------------------------------------

class TestPlotComparison:
    def test_creates_file(self, tmp_plot_dir):
        h1 = {"epoch": [1, 2, 3], "loss": [0.5, 0.3, 0.1]}
        h2 = {"epoch": [1, 2, 3], "loss": [0.6, 0.4, 0.2]}
        path = plot_comparison(
            [h1, h2], ["Model A", "Model B"],
            save_path=os.path.join(tmp_plot_dir, "comp.png"),
        )
        assert os.path.exists(path)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            plot_comparison([{"loss": [1]}], ["A", "B"])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            plot_comparison([], [])

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="no data"):
            plot_comparison(
                [{"epoch": [1], "loss": [0.1]}, {"epoch": [1]}],
                ["A", "B"],
            )


# ---------------------------------------------------------------------------
# 6. plot_accuracy
# ---------------------------------------------------------------------------

class TestPlotAccuracy:
    def test_creates_file(self, xor_history, tmp_plot_dir):
        path = plot_accuracy(
            xor_history,
            save_path=os.path.join(tmp_plot_dir, "acc.png"),
        )
        assert os.path.exists(path)

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="no data"):
            plot_accuracy({"epoch": [1]})


# ---------------------------------------------------------------------------
# 7. Output directory handling
# ---------------------------------------------------------------------------

class TestOutputDirectory:
    def test_creates_nested_dir(self, tmp_plot_dir):
        nested = os.path.join(tmp_plot_dir, "a", "b", "c")
        path = plot_loss({"loss": [0.5]}, save_path=os.path.join(nested, "loss.png"))
        assert os.path.exists(path)

    def test_default_dir_created(self):
        # plot_loss with no save_path should create artifacts/plots/
        path = plot_loss({"loss": [0.5]})
        assert os.path.exists(path)
        assert "artifacts" in path
        os.remove(path)
