"""
Neural-network layers built on the scalar autodiff engine.

This module provides the core building blocks for neural networks:

- Module: Base class for layers with forward, parameters, zero_grad
- Neuron: Single neuron (weighted sum + bias)
- Linear: Fully-connected layer (nin inputs → nout outputs)
- ReLU: ReLU activation layer

All layers operate on scalar Values and use the existing autodiff engine.
No new mathematical operations are introduced — layers compose existing
Value operations (addition, multiplication, relu, tanh).

Design Decisions:
    D28: Module is a lightweight base class (not abstract).
    D29: Linear uses direct weight/bias Parameters.
    D30: ReLU/Tanh wrap existing Value operations.
"""

from __future__ import annotations

import math
from typing import List

from neuralearn.value import Value
from neuralearn.parameter import Parameter


class Module:
    """Base class for neural-network layers.

    Provides three core interfaces:
    - forward(*args): Compute output from inputs
    - parameters(): Discover all trainable Parameters
    - zero_grad(): Reset all parameter gradients to zero

    Subclasses must implement forward() and parameters().
    zero_grad() has a default implementation using parameters().
    """

    def __call__(self, *args):
        """Delegate to forward(). Makes module instances callable."""
        return self.forward(*args)

    def forward(self, *args) -> Value:
        """Compute the forward pass. Subclasses must override."""
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        """Return all trainable parameters in this module.

        Subclasses must override to return their parameters.
        """
        return []

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to zero."""
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Neuron(Module):
    """A single neuron: weighted sum + bias.

    Computes:
        y = w1*x1 + w2*x2 + ... + wn*xn + b

    where w_i and b are trainable Parameters.

    Args:
        nin: Number of input connections.

    Example:
        n = Neuron(3)
        x = [Value(1.0), Value(2.0), Value(3.0)]
        y = n(x)  # y = w1*1 + w2*2 + w3*3 + b
    """

    def __init__(self, nin: int) -> None:
        """Initialize a neuron with nin input weights and 1 bias.

        Weights are initialized from N(0, 1/sqrt(nin)).
        Bias is initialized to 0.

        Args:
            nin: Number of input connections.
        """
        if nin < 1:
            raise ValueError(f"Neuron requires at least 1 input, got {nin}")

        scale = 1.0 / math.sqrt(nin)
        self.weights: List[Parameter] = [
            Parameter(scale * (2 * (i * 7 + 3) % 11 / 11.0 - 1))
            for i in range(nin)
        ]
        self.bias: Parameter = Parameter(0.0)

    def forward(self, x: List[Value | float]) -> Value:
        """Compute weighted sum + bias.

        Args:
            x: List of nin input Values or scalars.

        Returns:
            A Value representing w1*x1 + ... + wn*xn + b.

        Raises:
            ValueError: If input length doesn't match nin.
        """
        if len(x) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} inputs, got {len(x)}"
            )

        # w1*x1 + w2*x2 + ... + wn*xn
        out = self.weights[0] * x[0]
        for wi, xi in zip(self.weights[1:], x[1:]):
            out = out + wi * xi

        # + bias
        out = out + self.bias
        return out

    def parameters(self) -> List[Parameter]:
        """Return all weights and the bias."""
        return self.weights + [self.bias]

    def __repr__(self) -> str:
        return f"Neuron(nin={len(self.weights)})"


class Linear(Module):
    """Fully-connected layer: nin inputs → nout outputs.

    Each output j computes:
        y_j = w_{j,1}*x_1 + w_{j,2}*x_2 + ... + w_{j,nin}*x_nin + b_j

    Total parameters: nout * (nin + 1)  (nout weight vectors + nout biases)

    Args:
        nin: Number of input features.
        nout: Number of output features.

    Example:
        layer = Linear(3, 2)
        x = [Value(1.0), Value(2.0), Value(3.0)]
        y = layer(x)  # y is a list of 2 Values
    """

    def __init__(self, nin: int, nout: int) -> None:
        """Initialize the linear layer.

        Args:
            nin: Number of input features.
            nout: Number of output features.
        """
        if nin < 1:
            raise ValueError(f"Linear requires at least 1 input, got {nin}")
        if nout < 1:
            raise ValueError(f"Linear requires at least 1 output, got {nout}")

        self.nin = nin
        self.nout = nout
        self.neurons: List[Neuron] = [Neuron(nin) for _ in range(nout)]

    def forward(self, x: List[Value | float]) -> List[Value]:
        """Compute forward pass through all neurons.

        Args:
            x: List of nin input Values or scalars.

        Returns:
            List of nout output Values, one per neuron.

        Raises:
            ValueError: If input length doesn't match nin.
        """
        if len(x) != self.nin:
            raise ValueError(
                f"Expected {self.nin} inputs, got {len(x)}"
            )

        return [neuron.forward(x) for neuron in self.neurons]

    def parameters(self) -> List[Parameter]:
        """Return all parameters from all neurons (weights + biases).

        Returns:
            Flat list of all Parameters in deterministic order.
        """
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

    def __repr__(self) -> str:
        return f"Linear(nin={self.nin}, nout={self.nout})"


class ReLU(Module):
    """ReLU activation layer: applies max(0, x) element-wise.

    Wraps the existing Value.relu() operation.
    No new mathematics is introduced.

    Args:
        Nothing (stateless activation).

    Example:
        act = ReLU()
        y = act(Value(-1.0))  # y = 0.0
        y = act(Value(2.0))   # y = 2.0
    """

    def forward(self, x: Value) -> Value:
        """Apply ReLU activation.

        Args:
            x: Input Value.

        Returns:
            max(0, x) as a Value.
        """
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Tanh(Module):
    """Tanh activation layer: applies tanh(x) element-wise.

    Wraps the existing Value.tanh() operation.
    No new mathematics is introduced.

    Example:
        act = Tanh()
        y = act(Value(0.0))  # y = 0.0
        y = act(Value(1.0))  # y ≈ 0.762
    """

    def forward(self, x: Value) -> Value:
        """Apply tanh activation.

        Args:
            x: Input Value.

        Returns:
            tanh(x) as a Value.
        """
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"


class Sigmoid(Module):
    """Sigmoid activation layer: applies sigmoid(x) = 1/(1+exp(-x)) element-wise.

    Wraps the existing Value.sigmoid() operation.
    No new mathematics is introduced.

    Example:
        act = Sigmoid()
        y = act(Value(0.0))  # y = 0.5
        y = act(Value(2.0))  # y ≈ 0.881
    """

    def forward(self, x: Value) -> Value:
        """Apply sigmoid activation.

        Args:
            x: Input Value.

        Returns:
            sigmoid(x) as a Value.
        """
        return x.sigmoid()

    def __repr__(self) -> str:
        return "Sigmoid()"
