"""
neuralearn - A neural network framework built from first principles.

This package provides an educational implementation of a deep learning framework,
demonstrating the fundamental machinery behind neural-network training.
"""

__version__ = "1.1.0"

from neuralearn.value import Value
from neuralearn.parameter import Parameter
from neuralearn.layers import Module, Neuron, Linear, ReLU, Tanh, Sigmoid
from neuralearn.losses import mse_loss, binary_cross_entropy
from neuralearn.optimizers import SGD, MomentumSGD, Adam
from neuralearn.training import Trainer
from neuralearn.datasets import Dataset
from neuralearn.dataloaders import DataLoader
from neuralearn.visualization import (
    plot_loss,
    plot_train_test_curve,
    plot_regression_fit,
    plot_xor_predictions,
    plot_comparison,
    plot_accuracy,
)

__all__ = [
    "Value", "Parameter", "Module", "Neuron", "Linear", "ReLU", "Tanh", "Sigmoid",
    "mse_loss", "binary_cross_entropy",
    "SGD", "MomentumSGD", "Adam",
    "Trainer",
    "Dataset", "DataLoader",
    "plot_loss", "plot_train_test_curve", "plot_regression_fit",
    "plot_xor_predictions", "plot_comparison", "plot_accuracy",
]
