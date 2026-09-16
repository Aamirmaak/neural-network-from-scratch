"""
neuralearn - A neural network framework built from first principles.

This package provides an educational implementation of a deep learning framework,
demonstrating the fundamental machinery behind neural-network training.

Public API:
    Core:       Value, Parameter
    Layers:     Module, Neuron, Linear, ReLU, Tanh, Sigmoid
    Losses:     mse_loss, binary_cross_entropy
    Optimizers: SGD, MomentumSGD, Adam
    Training:   Trainer
    Data:       Dataset, DataLoader
    Validation: numerical_grad, gradient_check
    Viz (opt):  plot_loss, plot_train_test_curve, plot_regression_fit,
                plot_xor_predictions, plot_comparison, plot_accuracy
"""

__version__ = "1.4.0"

from neuralearn.value import Value
from neuralearn.parameter import Parameter
from neuralearn.layers import Module, Neuron, Linear, ReLU, Tanh, Sigmoid
from neuralearn.losses import mse_loss, binary_cross_entropy
from neuralearn.optimizers import SGD, MomentumSGD, Adam
from neuralearn.training import Trainer
from neuralearn.datasets import Dataset
from neuralearn.dataloaders import DataLoader
from neuralearn.gradient_check import numerical_grad, gradient_check

__all__ = [
    # Core
    "Value", "Parameter",
    # Layers
    "Module", "Neuron", "Linear", "ReLU", "Tanh", "Sigmoid",
    # Losses
    "mse_loss", "binary_cross_entropy",
    # Optimizers
    "SGD", "MomentumSGD", "Adam",
    # Training
    "Trainer",
    # Data
    "Dataset", "DataLoader",
    # Validation
    "numerical_grad", "gradient_check",
]


def plot_loss(*args, **kwargs):  # noqa: D401
    """Lazy-loaded visualization — requires matplotlib."""
    from neuralearn.visualization import plot_loss as _plot_loss
    return _plot_loss(*args, **kwargs)


def plot_train_test_curve(*args, **kwargs):
    """Lazy-loaded visualization — requires matplotlib."""
    from neuralearn.visualization import plot_train_test_curve as _fn
    return _fn(*args, **kwargs)


def plot_regression_fit(*args, **kwargs):
    """Lazy-loaded visualization — requires matplotlib."""
    from neuralearn.visualization import plot_regression_fit as _fn
    return _fn(*args, **kwargs)


def plot_xor_predictions(*args, **kwargs):
    """Lazy-loaded visualization — requires matplotlib."""
    from neuralearn.visualization import plot_xor_predictions as _fn
    return _fn(*args, **kwargs)


def plot_comparison(*args, **kwargs):
    """Lazy-loaded visualization — requires matplotlib."""
    from neuralearn.visualization import plot_comparison as _fn
    return _fn(*args, **kwargs)


def plot_accuracy(*args, **kwargs):
    """Lazy-loaded visualization — requires matplotlib."""
    from neuralearn.visualization import plot_accuracy as _fn
    return _fn(*args, **kwargs)
