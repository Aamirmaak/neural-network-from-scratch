"""
Scalar Value with reverse-mode automatic differentiation.

This module implements a scalar `Value` class that participates in a
computational graph. Each arithmetic operation creates a new Value node
with references to its parents and a local backward function. Calling
`backward()` on the output node propagates gradients through the graph
in reverse topological order.

Mathematical background:

    For a scalar function z = f(x, y, ...), reverse-mode autodiff computes
    dz/dx, dz/dy, ... by traversing the computational graph from z backward
    to each leaf variable, applying the chain rule at every node.

Stage 1 operations: +, *, neg (-x), -, **
Stage 2 operations: /, reciprocal, exp, log, tanh, relu
"""

from __future__ import annotations

import math
from typing import Callable, List, Set


class Value:
    """A scalar value participating in a computational graph.

    Attributes:
        data: The scalar numerical value.
        grad: Accumulated gradient with respect to the output (initialized to 0).
        _prev: Set of parent Value nodes in the computational graph.
        _op: Name of the operation that produced this Value (empty for leaves).
        _backward: Local backward function that propagates gradient to parents.
    """

    def __init__(
        self,
        data: float,
        _prev: Set[Value] | None = None,
        _op: str = "",
        _backward: Callable[[], None] | None = None,
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = _prev if _prev is not None else set()
        self._op = _op
        self._backward = _backward if _backward is not None else lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    # ------------------------------------------------------------------
    # Forward operations
    # ------------------------------------------------------------------

    def __add__(self, other: Value | float) -> Value:
        """z = x + y

        Local derivatives:
            dz/dx = 1
            dz/dy = 1

        Backward rule:
            dx += dout * 1
            dy += dout * 1
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data + other.data,
            _prev={self, other},
            _op="+",
        )

        def _backward() -> None:
            # dz/dx = 1, dz/dy = 1
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> Value:
        """Support scalar + Value (reflected addition)."""
        return self.__add__(other)

    def __mul__(self, other: Value | float) -> Value:
        """z = x * y

        Local derivatives:
            dz/dx = y
            dz/dy = x

        Backward rule:
            dx += dout * y
            dy += dout * x
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data * other.data,
            _prev={self, other},
            _op="*",
        )

        def _backward() -> None:
            # dz/dx = y, dz/dy = x
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> Value:
        """Support scalar * Value (reflected multiplication)."""
        return self.__mul__(other)

    def __neg__(self) -> Value:
        """z = -x

        Local derivative:
            dz/dx = -1

        Backward rule:
            dx += dout * (-1)
        """
        out = Value(
            -self.data,
            _prev={self},
            _op="neg",
        )

        def _backward() -> None:
            self.grad += out.grad * (-1.0)

        out._backward = _backward
        return out

    def __sub__(self, other: Value | float) -> Value:
        """z = x - y

        Implemented as x + (-y). Local derivatives:
            dz/dx = 1
            dz/dy = -1

        Backward rule:
            dx += dout * 1
            dy += dout * (-1)
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data - other.data,
            _prev={self, other},
            _op="-",
        )

        def _backward() -> None:
            # dz/dx = 1, dz/dy = -1
            self.grad += out.grad * 1.0
            other.grad += out.grad * (-1.0)

        out._backward = _backward
        return out

    def __rsub__(self, other: float) -> Value:
        """Support scalar - Value (reflected subtraction)."""
        other_val = other if isinstance(other, Value) else Value(other)
        return other_val.__sub__(self)

    def __pow__(self, other: float) -> Value:
        """z = x^n  (scalar power, n must be a Python int or float)

        Local derivative:
            dz/dx = n * x^(n-1)

        Backward rule:
            dx += dout * n * x^(n-1)

        Domain assumption: x > 0 when n is not an integer, to avoid
        complex-valued results. No explicit check is enforced; the
        caller is responsible for providing valid inputs.
        """
        if not isinstance(other, (int, float)):
            raise TypeError("Power exponent must be a numeric scalar")

        n = float(other)
        out = Value(
            self.data ** n,
            _prev={self},
            _op="**",
        )

        def _backward() -> None:
            # dz/dx = n * x^(n-1)
            self.grad += out.grad * n * (self.data ** (n - 1.0))

        out._backward = _backward
        return out

    def __truediv__(self, other: Value | float) -> Value:
        """z = x / y

        Local derivatives:
            dz/dx = 1/y
            dz/dy = -x/y²

        Backward rule:
            dx += dout * (1/y)
            dy += dout * (-x/y²)
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data / other.data,
            _prev={self, other},
            _op="/",
        )

        def _backward() -> None:
            # dz/dx = 1/y, dz/dy = -x/y²
            self.grad += out.grad * (1.0 / other.data)
            other.grad += out.grad * (-self.data / (other.data ** 2))

        out._backward = _backward
        return out

    def __rtruediv__(self, other: float) -> Value:
        """Support scalar / Value (reflected division)."""
        other_val = other if isinstance(other, Value) else Value(other)
        return other_val.__truediv__(self)

    def exp(self) -> Value:
        """z = exp(x)

        Local derivative:
            dz/dx = exp(x)

        Backward rule:
            dx += dout * exp(x)
        """
        out = Value(
            math.exp(self.data),
            _prev={self},
            _op="exp",
        )

        def _backward() -> None:
            # dz/dx = exp(x)
            self.grad += out.grad * math.exp(self.data)

        out._backward = _backward
        return out

    def log(self) -> Value:
        """z = log(x)  (natural logarithm)

        Local derivative:
            dz/dx = 1/x

        Backward rule:
            dx += dout * (1/x)

        Domain: x > 0. No explicit check is enforced; the caller
        is responsible for providing valid inputs.
        """
        out = Value(
            math.log(self.data),
            _prev={self},
            _op="log",
        )

        def _backward() -> None:
            # dz/dx = 1/x
            self.grad += out.grad * (1.0 / self.data)

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """z = tanh(x)

        Local derivative:
            dz/dx = 1 - tanh(x)²

        Backward rule:
            dx += dout * (1 - tanh(x)²)
        """
        t = math.tanh(self.data)
        out = Value(
            t,
            _prev={self},
            _op="tanh",
        )

        def _backward() -> None:
            # dz/dx = 1 - tanh(x)²
            self.grad += out.grad * (1.0 - t ** 2)

        out._backward = _backward
        return out

    def relu(self) -> Value:
        """z = max(0, x)  (Rectified Linear Unit)

        Local derivative:
            dz/dx = 1  if x > 0
            dz/dx = 0  if x <= 0

        Backward rule:
            dx += dout * (1 if x > 0 else 0)

        Convention: gradient is 0 at x = 0 (standard in deep learning).
        """
        out = Value(
            max(0.0, self.data),
            _prev={self},
            _op="relu",
        )

        def _backward() -> None:
            # dz/dx = 1 if x > 0, else 0
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)

        out._backward = _backward
        return out

    def reciprocal(self) -> Value:
        """z = 1/x  (Reciprocal)

        Local derivative:
            dz/dx = -1/x²

        Backward rule:
            dx += dout * (-1/x²)
        """
        out = Value(
            1.0 / self.data,
            _prev={self},
            _op="1/",
        )

        def _backward() -> None:
            # dz/dx = -1/x²
            self.grad += out.grad * (-1.0 / (self.data ** 2))

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self) -> None:
        """Propagate gradients backward through the computational graph.

        Semantic model (accumulation):

            Each call to backward() computes the gradient of this output
            with respect to every node in the graph and **adds** the result
            to the existing leaf-node gradients.  Intermediate (non-leaf)
            gradients are freshly computed on every call so that stale
            values from a prior call never corrupt the current traversal.

            Root gradient: always set to 1.0 per call (d(output)/d(output)).
            Leaf gradients: accumulate across calls (+= semantics).
            Intermediate gradients: reset to 0.0 before each traversal.

        Algorithm:
            1. Build topological order by DFS from self
            2. Reset gradients:
                 - root  → 1.0
                 - intermediate (non-root, non-leaf) → 0.0
                 - leaf  → untouched (accumulate)
            3. For each node in REVERSE topological order:
                 node._backward()
                   → accumulates gradient contributions into each parent

        Topological ordering ensures that when we process a node,
        the gradient flowing into it from all downstream paths has
        already been fully accumulated.
        """
        # Build topological order via DFS
        visited: Set[Value] = set()
        order: List[Value] = []

        def _topo_sort(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    _topo_sort(parent)
                order.append(node)

        _topo_sort(self)

        # Reset gradients for a fresh backward pass.
        # - Root node:        set to 1.0
        # - Intermediate nodes (have parents, non-leaf): reset to 0.0
        # - Leaf nodes (no parents): untouched — gradients accumulate across calls
        for node in order:
            if node is self:
                node.grad = 1.0
            elif len(node._prev) > 0:
                node.grad = 0.0

        # Traverse in reverse topological order (output → leaves)
        for node in reversed(order):
            node._backward()
