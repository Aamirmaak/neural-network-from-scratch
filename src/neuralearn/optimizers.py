"""
Optimizers for the scalar autodiff engine.

This module provides parameter-update optimizers suitable for
neural-network training:

- SGD: Stochastic Gradient Descent (vanilla)
- MomentumSGD: SGD with momentum
- Adam: Adaptive moment estimation

All optimizers:
- Accept a list of trainable Parameters
- Read parameter.grad to compute updates
- Modify parameter.data in-place
- Preserve Parameter identity (no replacement)
- Support zero_grad() to reset accumulated gradients

Design Decisions:
    D34: Optimizers are classes (not functions) because they maintain
         per-parameter state (velocity for Momentum, moments for Adam).
         SGD could be a function, but class-based API is consistent.
    D35: Optimizers accept List[Parameter] directly, not Module objects.
         This keeps optimizers decoupled from model architecture.
    D36: Parameter state is keyed by id(parameter), which is stable
         for the lifetime of the Parameter object.
    D37: optimizer.zero_grad() delegates to Parameter.zero_grad().
"""

from __future__ import annotations

import math
from typing import Dict, List

from neuralearn.parameter import Parameter


class SGD:
    """Stochastic Gradient Descent optimizer.

    Update rule for each parameter theta with gradient g:

        theta := theta - lr * g

    Args:
        parameters: List of trainable Parameters to update.
        lr: Learning rate (must be > 0).

    Example:
        >>> p = Parameter(5.0)
        >>> opt = SGD([p], lr=0.1)
        >>> # ... compute loss, call loss.backward() ...
        >>> p.grad = 2.0
        >>> opt.step()
        >>> p.data  # 5.0 - 0.1 * 2.0 = 4.8
        4.8
    """

    def __init__(self, parameters: List[Parameter], lr: float = 0.01) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.parameters = list(parameters)
        self.lr = lr

    def step(self) -> None:
        """Apply gradient update to all parameters.

        For each parameter: theta := theta - lr * grad.
        """
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to zero.

        Delegates to Parameter.zero_grad() for each parameter.
        """
        for p in self.parameters:
            p.zero_grad()

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, n_params={len(self.parameters)})"


class MomentumSGD:
    """SGD with momentum.

    Update rule for each parameter theta with gradient g:

        v_t = momentum * v_{t-1} + g_t
        theta_t = theta_{t-1} - lr * v_t

    Initial velocity: v_0 = 0

    Args:
        parameters: List of trainable Parameters to update.
        lr: Learning rate (must be > 0).
        momentum: Momentum coefficient (typically 0.9, must be >= 0).

    Example:
        >>> p = Parameter(5.0)
        >>> opt = MomentumSGD([p], lr=0.1, momentum=0.9)
        >>> p.grad = 2.0
        >>> opt.step()  # v = 0.9*0 + 2 = 2; theta -= 0.1*2 = 4.8
        >>> p.data
        4.8
    """

    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if momentum < 0:
            raise ValueError(f"Momentum must be >= 0, got {momentum}")
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        # Per-parameter velocity, keyed by parameter id
        self._velocity: Dict[int, float] = {}

    def step(self) -> None:
        """Apply momentum gradient update to all parameters.

        For each parameter:
            v = momentum * v + grad
            theta -= lr * v
        """
        for p in self.parameters:
            pid = id(p)
            v = self._velocity.get(pid, 0.0)
            v = self.momentum * v + p.grad
            self._velocity[pid] = v
            p.data -= self.lr * v

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to zero.

        Does NOT reset velocity state — momentum persists across steps.
        """
        for p in self.parameters:
            p.zero_grad()

    def __repr__(self) -> str:
        return (
            f"MomentumSGD(lr={self.lr}, momentum={self.momentum}, "
            f"n_params={len(self.parameters)})"
        )


class Adam:
    """Adam optimizer (Adaptive Moment Estimation).

    Update rule for each parameter theta with gradient g:

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat_t = m_t / (1 - beta1^t)
        v_hat_t = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)

    Initial: m_0 = 0, v_0 = 0, t = 0

    Args:
        parameters: List of trainable Parameters to update.
        lr: Learning rate (must be > 0).
        beta1: Exponential decay rate for first moment (default 0.9).
        beta2: Exponential decay rate for second moment (default 0.999).
        epsilon: Small constant for numerical stability (default 1e-8).

    Example:
        >>> p = Parameter(5.0)
        >>> opt = Adam([p], lr=0.001)
        >>> p.grad = 2.0
        >>> opt.step()  # First step: bias-corrected moments
    """

    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Per-parameter state, keyed by parameter id
        self._m: Dict[int, float] = {}  # First moment
        self._v: Dict[int, float] = {}  # Second moment
        self._t: Dict[int, int] = {}    # Timestep

    def step(self) -> None:
        """Apply Adam update to all parameters.

        For each parameter:
            m = beta1*m + (1-beta1)*grad
            v = beta2*v + (1-beta2)*grad^2
            m_hat = m / (1 - beta1^t)
            v_hat = v / (1 - beta2^t)
            theta -= lr * m_hat / (sqrt(v_hat) + eps)
        """
        for p in self.parameters:
            pid = id(p)

            # Initialize state for new parameters
            if pid not in self._t:
                self._m[pid] = 0.0
                self._v[pid] = 0.0
                self._t[pid] = 0

            # Increment timestep
            self._t[pid] += 1
            t = self._t[pid]

            # Update biased moments
            self._m[pid] = self.beta1 * self._m[pid] + (1 - self.beta1) * p.grad
            self._v[pid] = self.beta2 * self._v[pid] + (1 - self.beta2) * p.grad * p.grad

            # Bias correction
            m_hat = self._m[pid] / (1 - self.beta1 ** t)
            v_hat = self._v[pid] / (1 - self.beta2 ** t)

            # Update parameter
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to zero.

        Does NOT reset optimizer state (moments, timestep) — these
        persist across steps and are part of the optimizer's memory.
        """
        for p in self.parameters:
            p.zero_grad()

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, "
            f"eps={self.epsilon}, n_params={len(self.parameters)})"
        )
