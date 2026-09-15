"""
Loss functions for the scalar autodiff engine.

This module provides differentiable loss functions suitable for
neural-network training:

- mse_loss: Mean Squared Error (regression)
- binary_cross_entropy: Binary Cross-Entropy (binary classification)

All loss functions:
- Accept Python sequences of Value-compatible scalars
- Return a scalar Value connected to the computational graph
- Support gradient flow back to predictions (and ultimately to Parameters)
- Use mean reduction (average over samples)

Design Decisions:
    D31: Loss functions are plain functions, not Module subclasses.
         They have no trainable parameters, no state, and no need for
         zero_grad(). Plain functions are simpler and more appropriate.
    D32: BCE clips probabilities to [eps, 1-eps] for numerical stability,
         using existing relu() operation: clip = eps + (p-eps).relu() - (p-(1-eps)).relu()
    D33: Both MSE and BCE use mean reduction by default.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from neuralearn.value import Value


def mse_loss(
    predictions: Sequence[Value | float],
    targets: Sequence[Value | float],
) -> Value:
    """Mean Squared Error loss with mean reduction.

    Computes:
        MSE = (1/n) * sum((y_hat_i - y_i)^2)

    where n is the number of samples, y_hat_i are predictions, y_i are targets.

    The returned Value is part of the computational graph. Calling backward()
    on it propagates gradients to prediction values.

    Mathematical gradient:
        dMSE/dy_hat_i = 2(y_hat_i - y_i) / n

    Args:
        predictions: Sequence of predicted values (Value or float).
        targets: Sequence of target values (Value or float).
            Targets are treated as constants (gradients don't flow through them).

    Returns:
        A scalar Value equal to the mean squared error.

    Raises:
        ValueError: If predictions and targets have different lengths.
        ValueError: If inputs are empty.

    Example:
        >>> from neuralearn.value import Value
        >>> loss = mse_loss([Value(1.0), Value(2.0)], [Value(1.5), Value(3.0)])
        >>> loss.backward()
        # loss.data = ((1.0-1.5)^2 + (2.0-3.0)^2) / 2 = 0.625
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have same length, "
            f"got {len(predictions)} and {len(targets)}"
        )
    if len(predictions) == 0:
        raise ValueError("mse_loss requires at least one prediction-target pair")

    n = len(predictions)

    # Accumulate (y_hat_i - y_i)^2 across samples
    total = predictions[0] - targets[0]
    total = total * total
    for pred, target in zip(predictions[1:], targets[1:]):
        diff = pred - target
        total = total + diff * diff

    # Mean reduction: divide by n
    return total * (1.0 / n)


def binary_cross_entropy(
    predictions: Sequence[Value | float],
    targets: Sequence[Value | float],
    eps: float = 1e-7,
) -> Value:
    """Binary Cross-Entropy loss with mean reduction.

    Computes:
        BCE = -(1/n) * sum[y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]

    where n is the number of samples, p_i are predicted probabilities,
    and y_i in {0, 1} are binary targets.

    Numerical Stability:
        Predictions are clipped to [eps, 1-eps] to prevent log(0).
        Default eps=1e-7 means predictions are clamped to [1e-7, 1-1e-7].
        Clipping uses existing Value.relu() operation:
            clip(p) = eps + (p - eps).relu() - (p - (1-eps)).relu()
        This is differentiable everywhere except at the exact boundary points.
        For typical predictions away from 0 and 1, gradients flow correctly.

    Mathematical gradient (for valid 0 < p_i < 1):
        dBCE/dp_i = (1/n) * [-y_i/p_i + (1-y_i)/(1-p_i)]

    Args:
        predictions: Sequence of predicted probabilities (Value or float).
            Values should be in (0, 1). Values outside this range are clipped.
        targets: Sequence of binary targets (Value or float).
            Values should be in {0, 1}. No enforcement is applied.
        eps: Clipping margin for numerical stability (default 1e-7).

    Returns:
        A scalar Value equal to the mean binary cross-entropy.

    Raises:
        ValueError: If predictions and targets have different lengths.
        ValueError: If inputs are empty.

    Example:
        >>> from neuralearn.value import Value
        >>> loss = binary_cross_entropy([Value(0.9)], [Value(1.0)])
        >>> loss.backward()
        # loss.data = -(1.0*log(0.9) + 0.0*log(0.1)) ~ 0.105
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have same length, "
            f"got {len(predictions)} and {len(targets)}"
        )
    if len(predictions) == 0:
        raise ValueError(
            "binary_cross_entropy requires at least one prediction-target pair"
        )

    n = len(predictions)
    one_minus_eps = 1.0 - eps

    total = Value(0.0)
    for pred, target in zip(predictions, targets):
        p = pred if isinstance(pred, Value) else Value(pred)
        y = target if isinstance(target, Value) else Value(target)

        # Clip prediction to [eps, 1-eps] using relu:
        # clip(p) = eps + max(0, p-eps) - max(0, p-(1-eps))
        clipped = Value(eps) + (p - Value(eps)).relu() - (p - Value(one_minus_eps)).relu()

        # BCE term: -[y*log(p) + (1-y)*log(1-p)]
        term = y * clipped.log() + (Value(1.0) - y) * (Value(1.0) - clipped).log()
        total = total + term

    # Mean reduction and negation
    return total * (-1.0 / n)
