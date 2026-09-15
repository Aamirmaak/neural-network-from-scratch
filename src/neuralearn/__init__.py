"""
neuralearn - A neural network framework built from first principles.

This package provides an educational implementation of a deep learning framework,
demonstrating the fundamental machinery behind neural-network training.
"""

__version__ = "0.8.0"

from neuralearn.value import Value
from neuralearn.parameter import Parameter
from neuralearn.layers import Module, Neuron, Linear, ReLU, Tanh
from neuralearn.losses import mse_loss, binary_cross_entropy
from neuralearn.optimizers import SGD, MomentumSGD, Adam
from neuralearn.training import Trainer

__all__ = [
    "Value", "Parameter", "Module", "Neuron", "Linear", "ReLU", "Tanh",
    "mse_loss", "binary_cross_entropy",
    "SGD", "MomentumSGD", "Adam",
    "Trainer",
]
