"""
Trainable parameter for the scalar autodiff engine.

A Parameter is a Value that represents a trainable scalar quantity.
It extends Value with gradient-tracking semantics (requires_grad)
and provides zero_grad() for resetting accumulated gradients.

Design Decision (D27): Parameter subclasses Value. This IS-A relationship
means Parameter naturally participates in the existing computational graph.
All Value arithmetic operations (addition, multiplication, etc.) work
directly on Parameters without adapter code.
"""

from __future__ import annotations

from neuralearn.value import Value


class Parameter(Value):
    """A trainable scalar value extending Value.

    Attributes:
        data: The scalar numerical value (inherited from Value).
        grad: Accumulated gradient (inherited from Value).
        requires_grad: Whether this parameter should be trained (default True).

    Example:
        w = Parameter(0.5)
        x = Value(2.0)
        out = w * x       # participates in computational graph
        out.backward()     # w.grad is now populated
        w.zero_grad()      # w.grad reset to 0.0
    """

    def __init__(self, data: float, requires_grad: bool = True) -> None:
        """Initialize a trainable parameter.

        Args:
            data: Initial scalar value.
            requires_grad: Whether this parameter is trainable (default True).
        """
        super().__init__(data)
        self.requires_grad = requires_grad

    def zero_grad(self) -> None:
        """Reset the accumulated gradient to zero.

        This is an explicit operation — computational graph intermediate
        gradients are managed by backward(), but parameter gradients
        accumulate across backward calls and must be explicitly zeroed.
        """
        self.grad = 0.0

    def __repr__(self) -> str:
        return f"Parameter(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"
