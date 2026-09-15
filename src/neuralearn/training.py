"""
Training engine for the scalar autodiff neural-network framework.

This module provides the Trainer class, which connects a model (Module),
a loss function, and an optimizer into a complete training lifecycle.

Training lifecycle per sample:
    1. Forward pass:    prediction = model(input)
    2. Loss computation: loss = loss_fn(prediction, target)
    3. Backward pass:    loss.backward()
    4. Parameter update: optimizer.step()
    5. Gradient reset:   optimizer.zero_grad()

Design Decisions:
    D38: Trainer is a class that accepts model, loss_fn, optimizer at construction.
    D39: Per-sample training (no batching) — consistent with scalar architecture.
    D40: Training data as parallel lists of inputs and targets.
    D41: History records per-epoch mean loss.
    D42: Evaluation computes loss without modifying parameters.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

from neuralearn.value import Value
from neuralearn.parameter import Parameter


class Trainer:
    """Training engine connecting model, loss function, and optimizer.

    Orchestrates the training lifecycle: forward → loss → backward →
    step → zero_grad, repeated for a configurable number of epochs.

    Args:
        model: A Module (or any object with __call__, parameters(), zero_grad()).
        loss_fn: A loss function accepting (predictions, targets) → Value.
        optimizer: An optimizer with step() and zero_grad() methods.

    Example:
        >>> from neuralearn.layers import Linear
        >>> from neuralearn.losses import mse_loss
        >>> from neuralearn.optimizers import SGD
        >>> model = Linear(2, 1)
        >>> optimizer = SGD(model.parameters(), lr=0.01)
        >>> trainer = Trainer(model, mse_loss, optimizer)
        >>> history = trainer.fit(
        ...     inputs=[[Value(1.0), Value(2.0)], [Value(3.0), Value(4.0)]],
        ...     targets=[[Value(5.0)], [Value(11.0)]],
        ...     epochs=100,
        ... )
        >>> len(history["loss"])
        100
    """

    def __init__(
        self,
        model: Any,
        loss_fn: Callable[..., Value],
        optimizer: Any,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Must have __call__ (forward), parameters(), and zero_grad().
            loss_fn: Must accept (predictions, targets) and return a scalar Value.
            optimizer: Must have step() and zero_grad() methods.

        Raises:
            TypeError: If model, loss_fn, or optimizer is None.
        """
        if model is None:
            raise TypeError("model must not be None")
        if loss_fn is None:
            raise TypeError("loss_fn must not be None")
        if optimizer is None:
            raise TypeError("optimizer must not be None")

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def fit(
        self,
        inputs: Sequence[Sequence[Any]],
        targets: Sequence[Any],
        epochs: int = 100,
    ) -> Dict[str, List[float]]:
        """Train the model on the provided data.

        For each epoch, iterates over all samples and performs:
            forward → loss → backward → optimizer.step → zero_grad

        Args:
            inputs: List of input sequences. Each input is passed to model().
                Example: [[Value(1.0), Value(2.0)], [Value(3.0), Value(4.0)]]
            targets: List of target values/sequences. Each target is passed
                to loss_fn alongside the model's prediction.
                Example: [Value(5.0)] or [[Value(5.0)], [Value(11.0)]]
            epochs: Number of training epochs (must be >= 1).

        Returns:
            Dictionary with key "loss" mapping to a list of per-epoch
            mean loss values (ordinary Python floats).

        Raises:
            ValueError: If epochs < 1.
            ValueError: If inputs and targets have different lengths.
            ValueError: If inputs is empty.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if len(inputs) == 0:
            raise ValueError("inputs must not be empty")
        if len(inputs) != len(targets):
            raise ValueError(
                f"inputs and targets must have same length, "
                f"got {len(inputs)} and {len(targets)}"
            )

        n_samples = len(inputs)
        history: Dict[str, List[float]] = {"loss": []}

        for epoch in range(epochs):
            epoch_loss = 0.0

            for x, y in zip(inputs, targets):
                # 1. Forward pass
                prediction = self.model(x)

                # 2. Loss computation
                loss = self.loss_fn(prediction, y)

                # 3. Backward pass
                loss.backward()

                # 4. Parameter update
                self.optimizer.step()

                # 5. Gradient reset
                self.optimizer.zero_grad()

                # Accumulate loss for epoch mean
                epoch_loss += loss.data

            # Record mean loss for this epoch
            history["loss"].append(epoch_loss / n_samples)

        return history

    def evaluate(
        self,
        inputs: Sequence[Sequence[Any]],
        targets: Sequence[Any],
    ) -> Dict[str, List[float]]:
        """Evaluate the model on the provided data without updating parameters.

        Performs forward passes and computes loss but does NOT call
        backward(), optimizer.step(), or zero_grad(). Parameters and
        optimizer state are unchanged.

        Args:
            inputs: List of input sequences (same format as fit()).
            targets: List of target values/sequences (same format as fit()).
            epochs: Number of evaluation passes (default 1). For single
                evaluation, use 1.

        Returns:
            Dictionary with key "loss" mapping to a list of per-pass
            mean loss values.

        Raises:
            ValueError: If inputs is empty.
            ValueError: If inputs and targets have different lengths.
        """
        if len(inputs) == 0:
            raise ValueError("inputs must not be empty")
        if len(inputs) != len(targets):
            raise ValueError(
                f"inputs and targets must have same length, "
                f"got {len(inputs)} and {len(targets)}"
            )

        n_samples = len(inputs)
        history: Dict[str, List[float]] = {"loss": []}
        total_loss = 0.0

        for x, y in zip(inputs, targets):
            # Forward pass only — no backward, no step, no zero_grad
            prediction = self.model(x)
            loss = self.loss_fn(prediction, y)
            total_loss += loss.data

        history["loss"].append(total_loss / n_samples)
        return history

    def __repr__(self) -> str:
        return (
            f"Trainer(model={self.model.__class__.__name__}, "
            f"loss_fn={self.loss_fn.__name__}, "
            f"optimizer={self.optimizer.__class__.__name__})"
        )
