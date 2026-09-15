"""
Visualization utilities for neural-network experiments.

Provides lightweight plotting functions for training analysis:
- Loss curves (training/test MSE, BCE)
- Regression fit visualization
- Classification prediction scatter
- Model/optimizer comparison plots

Dependency policy:
    matplotlib is an optional dependency. All functions raise ImportError
    with a clear message if matplotlib is not installed.

Output:
    All plots save to a specified directory (default: artifacts/plots/).
    No interactive GUI is required. Plots render correctly in headless
    environments using the Agg backend.

Design decisions:
    D49: Visualization is a separate analysis layer, not part of core framework.
    D50: matplotlib is the only visualization dependency.
    D51: Generated plots are committed to the repository for portfolio value.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence


def _ensure_matplotlib():
    """Import and configure matplotlib, raising a clear error if missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def _resolve_save_path(save_path: Optional[str], default_name: str) -> str:
    """Resolve save path, creating directories if needed."""
    if save_path is None:
        save_dir = os.path.join("artifacts", "plots")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, default_name)
    else:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    return save_path


def plot_loss(
    history: Dict[str, List[float]],
    title: str = "Training Loss",
    save_path: Optional[str] = None,
    loss_key: str = "loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
) -> str:
    """Plot training loss over epochs.

    Args:
        history: Dict with at least one key for x-axis data (default "loss")
            and optionally "epoch" for the x-axis. If "epoch" is absent,
            steps 1, 2, 3, ... are used.
        title: Plot title.
        save_path: File path to save the plot. If None, saves to
            artifacts/plots/<default_name>.
        loss_key: Key in history for the loss values.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Path to the saved plot file.
    """
    _, plt = _ensure_matplotlib()

    if loss_key not in history or len(history[loss_key]) == 0:
        raise ValueError(f"History has no data for key '{loss_key}'")

    epochs = history.get("epoch", list(range(1, len(history[loss_key]) + 1)))
    losses = history[loss_key]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, "b-o", markersize=3, linewidth=1.5, label=loss_key)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    save_path = _resolve_save_path(save_path, f"{loss_key}_curve.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


plot_train_test_loss = plot_loss  # alias for consistency


def plot_train_test_curve(
    history: Dict[str, List[float]],
    train_key: str = "train_mse",
    test_key: str = "test_mse",
    title: str = "Train vs Test Loss",
    save_path: Optional[str] = None,
) -> str:
    """Plot training and test loss on the same axes.

    Args:
        history: Dict with epoch, train_key, and test_key data.
        train_key: Key for training loss values.
        test_key: Key for test loss values.
        title: Plot title.
        save_path: File path to save.

    Returns:
        Path to the saved plot file.
    """
    _, plt = _ensure_matplotlib()

    if train_key not in history or len(history[train_key]) == 0:
        raise ValueError(f"History has no data for key '{train_key}'")
    if test_key not in history or len(history[test_key]) == 0:
        raise ValueError(f"History has no data for key '{test_key}'")

    epochs = history.get("epoch", list(range(1, len(history[train_key]) + 1)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history[train_key], "b-o", markersize=3, linewidth=1.5, label="Train")
    ax.plot(epochs, history[test_key], "r-s", markersize=3, linewidth=1.5, label="Test")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    save_path = _resolve_save_path(save_path, "train_test_loss.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_regression_fit(
    x_train: Sequence[float],
    y_train: Sequence[float],
    x_test: Sequence[float],
    y_test: Sequence[float],
    predictions: Sequence[float],
    target_fn=None,
    title: str = "Regression Fit",
    save_path: Optional[str] = None,
) -> str:
    """Plot regression data with model predictions and optional target function.

    Args:
        x_train: Training x values.
        y_train: Training target values.
        x_test: Test x values.
        y_test: Test target values.
        predictions: Model predictions for test x values.
        target_fn: Optional callable for the true function (plotted as a
            continuous curve if provided).
        title: Plot title.
        save_path: File path to save.

    Returns:
        Path to the saved plot file.
    """
    _, plt = _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot target function as a smooth curve
    if target_fn is not None:
        import math
        x_smooth = [x_test[0] + i * (x_test[-1] - x_test[0]) / 200
                     for i in range(201)]
        y_smooth = [target_fn(x) for x in x_smooth]
        ax.plot(x_smooth, y_smooth, "g--", linewidth=1.5, alpha=0.7, label="Target function")

    # Plot data points
    ax.scatter(x_train, y_train, c="blue", s=30, alpha=0.6, zorder=5, label="Train data")
    ax.scatter(x_test, y_test, c="red", s=30, alpha=0.6, zorder=5, label="Test data")

    # Plot predictions
    ax.scatter(x_test, predictions, c="orange", s=40, marker="x", linewidths=2,
               zorder=6, label="Model predictions")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    save_path = _resolve_save_path(save_path, "regression_fit.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_xor_predictions(
    results: List[Dict],
    title: str = "XOR Classification Predictions",
    save_path: Optional[str] = None,
) -> str:
    """Plot XOR classification predictions as a scatter.

    Shows input pairs colored by predicted class, with markers indicating
    correctness.

    Args:
        results: List of dicts with keys: input, target, prediction,
            predicted_class, correct.
        title: Plot title.
        save_path: File path to save.

    Returns:
        Path to the saved plot file.
    """
    _, plt = _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 6))

    for r in results:
        x, y = r["input"]
        color = "green" if r["correct"] else "red"
        marker = "o" if r["predicted_class"] == 1.0 else "s"
        edge = "darkgreen" if r["correct"] else "darkred"
        ax.scatter(x, y, c=color, s=200, marker=marker, edgecolors=edge,
                   linewidths=2, zorder=5)
        ax.annotate(f'{r["prediction"]:.3f}', (x, y),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color="black")

    # Draw decision regions (background shading)
    import numpy as np
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Input 1", fontsize=12)
    ax.set_ylabel("Input 2", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
               markersize=10, label="Correct (class 1)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="green",
               markersize=10, label="Correct (class 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=10, label="Incorrect (class 1)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
               markersize=10, label="Incorrect (class 0)"),
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

    save_path = _resolve_save_path(save_path, "xor_predictions.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_comparison(
    histories: List[Dict[str, List[float]]],
    labels: List[str],
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    loss_key: str = "loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
) -> str:
    """Plot multiple training curves on the same axes for comparison.

    Args:
        histories: List of history dicts, each containing loss_key data.
        labels: Labels for each curve.
        title: Plot title.
        save_path: File path to save.
        loss_key: Key in each history for loss values.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Path to the saved plot file.
    """
    _, plt = _ensure_matplotlib()

    if len(histories) != len(labels):
        raise ValueError("histories and labels must have same length")
    if len(histories) == 0:
        raise ValueError("Must provide at least one history")

    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    markers = ["o", "s", "^", "D", "v", "p"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (hist, label) in enumerate(zip(histories, labels)):
        if loss_key not in hist or len(hist[loss_key]) == 0:
            raise ValueError(f"History '{label}' has no data for key '{loss_key}'")
        epochs = hist.get("epoch", list(range(1, len(hist[loss_key]) + 1)))
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(epochs, hist[loss_key], f"-{marker}", color=color,
                markersize=3, linewidth=1.5, label=label)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    save_path = _resolve_save_path(save_path, "comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_accuracy(
    history: Dict[str, List[float]],
    title: str = "Training Accuracy",
    save_path: Optional[str] = None,
) -> str:
    """Plot accuracy over epochs.

    Args:
        history: Dict with "epoch" and "accuracy" keys.
        title: Plot title.
        save_path: File path to save.

    Returns:
        Path to the saved plot file.
    """
    _, plt = _ensure_matplotlib()

    if "accuracy" not in history or len(history["accuracy"]) == 0:
        raise ValueError("History has no data for key 'accuracy'")

    epochs = history.get("epoch", list(range(1, len(history["accuracy"]) + 1)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["accuracy"], "g-o", markersize=3, linewidth=1.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = _resolve_save_path(save_path, "accuracy_curve.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
