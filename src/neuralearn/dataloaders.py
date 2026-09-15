"""
DataLoader abstraction for the scalar autodiff neural-network framework.

This module provides a DataLoader class for iterating over a Dataset
in batches with optional shuffling.

Design Decisions:
    D44: DataLoader yields (batch_inputs, batch_targets) tuples.
    D45: Per-sample training within batches (D39 preserved).
    D46: Deterministic shuffling via explicit seed and dedicated Random instance.
    D47: Separate module from Dataset for clarity.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterator, List, Optional, Tuple

from neuralearn.datasets import Dataset


class DataLoader:
    """Iterates over a Dataset in batches with optional shuffling.

    Yields (batch_inputs, batch_targets) tuples where each is a list
    of individual samples. Supports configurable batch size, shuffling
    with deterministic seed, and drop_last for partial batches.

    Args:
        dataset: A Dataset instance to iterate over.
        batch_size: Number of samples per batch (must be >= 1).
        shuffle: Whether to shuffle samples before each iteration (default False).
        drop_last: Whether to drop the final incomplete batch (default False).
        seed: Random seed for deterministic shuffling (default None).

    Example:
        >>> ds = Dataset([[1, 2], [3, 4], [5, 6], [7, 8]], [10, 20, 30, 40])
        >>> loader = DataLoader(ds, batch_size=2, shuffle=True, seed=42)
        >>> for batch_inputs, batch_targets in loader:
        ...     print(batch_inputs, batch_targets)
        [3, 4], [7, 8] [20, 40]
        [1, 2], [5, 6] [10, 30]
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the DataLoader.

        Args:
            dataset: A Dataset instance.
            batch_size: Samples per batch (must be >= 1).
            shuffle: Whether to shuffle before each iteration.
            drop_last: Whether to discard the final incomplete batch.
            seed: Random seed for deterministic shuffling.

        Raises:
            TypeError: If dataset is None.
            ValueError: If batch_size < 1.
        """
        if dataset is None:
            raise TypeError("dataset must not be None")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        """Return the number of batches.

        Without drop_last: ceil(n / batch_size)
        With drop_last: floor(n / batch_size)
        """
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[Tuple[List[Any], List[Any]]]:
        """Iterate over the dataset in batches.

        If shuffle=True, generates a fresh permutation for each iteration
        using the internal RNG (seeded at construction).

        Yields:
            (batch_inputs, batch_targets) tuples where each is a list
            of individual samples.
        """
        n = len(self.dataset)

        if self.shuffle:
            indices = list(range(n))
            self._rng.shuffle(indices)
        else:
            indices = list(range(n))

        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if self.drop_last and end > n:
                break

            batch_indices = indices[start:end]
            batch_inputs = [self.dataset[i][0] for i in batch_indices]
            batch_targets = [self.dataset[i][1] for i in batch_indices]
            yield (batch_inputs, batch_targets)

    def __repr__(self) -> str:
        return (
            f"DataLoader(dataset={self.dataset!r}, "
            f"batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, "
            f"drop_last={self.drop_last})"
        )
