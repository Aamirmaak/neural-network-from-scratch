"""
Dataset abstraction for the scalar autodiff neural-network framework.

This module provides a lightweight Dataset class for storing paired
input/target data with validation.

Design Decisions:
    D43: Dataset stores references to input/target lists (no copying).
         Validates lengths match at construction. Supports Python sequence
         indexing semantics.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple


class Dataset:
    """A container for paired input/target data.

    Stores references to input and target sequences without copying.
    Validates that inputs and targets have the same length.

    Args:
        inputs: List of input sequences. Each element is passed to model().
            Example: [[Value(1.0), Value(2.0)], [Value(3.0), Value(4.0)]]
        targets: List of target values/sequences. Each element is passed
            to loss_fn alongside the model prediction.
            Example: [Value(5.0)] or [[Value(5.0)], [Value(11.0)]]

    Example:
        >>> ds = Dataset([[1.0, 2.0], [3.0, 4.0]], [5.0, 11.0])
        >>> len(ds)
        2
        >>> ds[0]
        ([1.0, 2.0], 5.0)
    """

    def __init__(
        self,
        inputs: Sequence[Any],
        targets: Sequence[Any],
    ) -> None:
        """Initialize the dataset.

        Args:
            inputs: List of input sequences.
            targets: List of target values/sequences.

        Raises:
            ValueError: If inputs and targets have different lengths.
            ValueError: If inputs is empty.
        """
        if len(inputs) == 0:
            raise ValueError("inputs must not be empty")
        if len(inputs) != len(targets):
            raise ValueError(
                f"inputs and targets must have same length, "
                f"got {len(inputs)} and {len(targets)}"
            )

        self._inputs = list(inputs)
        self._targets = list(targets)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._inputs)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return the (input, target) pair at index idx.

        Supports standard Python indexing including negative indices.

        Args:
            idx: Integer index.

        Returns:
            Tuple of (input, target) at the given index.

        Raises:
            IndexError: If idx is out of range.
        """
        return (self._inputs[idx], self._targets[idx])

    def __repr__(self) -> str:
        return f"Dataset(n_samples={len(self)})"
