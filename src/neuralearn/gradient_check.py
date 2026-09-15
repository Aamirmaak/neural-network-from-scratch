"""
Numerical gradient checking for the scalar autodiff engine.

This module provides utilities for verifying that analytical gradients
(computed by reverse-mode autodiff) match numerical approximations
computed using central-difference finite differences.

Mathematical background:

    The central-difference approximation to the derivative of f at x is:

        f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)

    This has O(ε²) truncation error, which is more accurate than the
    forward-difference approximation O(ε).

    Gradient checking compares the analytical gradient (from autodiff)
    against the numerical gradient (from finite differences) to detect
    bugs in backward function implementations.

Usage:

    from neuralearn.gradient_check import numerical_grad, gradient_check

    # Check gradient of a single input
    a = Value(2.0)
    out = a.exp()
    out.backward()
    num = numerical_grad(lambda: a.exp().data, a)
    assert abs(a.grad - num) < 1e-5

    # Check gradients of multiple inputs
    a, b = Value(2.0), Value(3.0)
    out = a * b
    out.backward()
    results = gradient_check(lambda: a.data * b.data, [a, b])
"""

from typing import Callable, List

from neuralearn.value import Value


def numerical_grad(fn: Callable[[], float], x_val: Value, eps: float = 1e-5) -> float:
    """Compute numerical gradient of fn with respect to a Value's data.

    Uses central difference: [f(x+ε) - f(x-ε)] / (2ε).

    The function fn must be a callable that returns a float when called.
    It should depend on x_val.data, which is perturbed during evaluation.

    Args:
        fn: Callable that returns a scalar. Should use x_val internally.
        x_val: The Value whose .data is perturbed to compute the gradient.
        eps: Perturbation size (default: 1e-5).

    Returns:
        Numerical approximation of df/dx at x_val.data.

    Notes:
        - fn is called twice: once with x_val.data + eps, once with x_val.data - eps.
        - x_val.data is restored to its original value after computation.
        - This function does NOT use autodiff; it relies purely on function evaluation.
        - The caller is responsible for ensuring fn depends on x_val.
    """
    original = x_val.data

    x_val.data = original + eps
    forward = fn()

    x_val.data = original - eps
    backward = fn()

    x_val.data = original  # restore
    return (forward - backward) / (2.0 * eps)


def gradient_check(
    fn: Callable[[], float],
    values: List[Value],
    eps: float = 1e-5,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> List[dict]:
    """Check analytical gradients against numerical approximations.

    For each Value in `values`, computes:
    1. The analytical gradient from fn's computational graph (via backward).
    2. The numerical gradient via central-difference finite differences.
    3. The absolute and relative errors between them.

    The function fn must:
    - Build a computational graph using the provided Value objects.
    - Return the output as a float.
    - The caller must have already called backward() on the output to populate
      the analytical gradients before calling this function.

    Args:
        fn: Callable that returns a scalar. Uses the Value objects in `values`.
        values: List of Value objects to check gradients for.
        eps: Perturbation size for finite differences (default: 1e-5).
        atol: Absolute tolerance for gradient comparison (default: 1e-5).
        rtol: Relative tolerance for gradient comparison (default: 1e-3).

    Returns:
        List of dicts, one per Value, containing:
            - 'value': the Value object
            - 'analytical': the analytical gradient (from backward)
            - 'numerical': the numerical gradient (from finite differences)
            - 'abs_error': absolute error
            - 'rel_error': relative error
            - 'passed': bool indicating whether the check passed

    Notes:
        - Numerical gradients are computed independently of autodiff.
        - Each Value is perturbed one at a time; other Values are untouched.
        - The function is called 2 * len(values) times for numerical gradient
          computation (central difference: 2 evaluations per value).
        - D19 semantics: analytical gradients must be computed by the caller
          before calling this function. This function reads existing .grad values.
    """
    results = []

    for val in values:
        # Read the analytical gradient (must already be computed by caller)
        analytical = val.grad

        # Compute numerical gradient via central difference
        original = val.data

        val.data = original + eps
        forward = fn()

        val.data = original - eps
        backward = fn()

        val.data = original  # restore

        numerical = (forward - backward) / (2.0 * eps)

        # Compute errors
        abs_error = abs(analytical - numerical)
        denom = max(abs(analytical), abs(numerical), 1e-12)
        rel_error = abs_error / denom

        passed = (abs_error < atol) or (rel_error < rtol)

        results.append({
            'value': val,
            'analytical': analytical,
            'numerical': numerical,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'passed': passed,
        })

    return results
