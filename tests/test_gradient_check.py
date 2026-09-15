"""
Comprehensive gradient-checking tests for the scalar autodiff engine.

This module tests the gradient_check utility and verifies that analytical
gradients (from reverse-mode autodiff) match numerical approximations
(central-difference finite differences) for all implemented operations.

Test categories:
    1. numerical_grad helper function
    2. gradient_check utility function
    3. Individual operation gradient checks
    4. Composed graph gradient checks
    5. Multi-input gradient checks
    6. Deep graph gradient checks
    7. Numerical edge cases and limitations
    8. Regression tests (D19, incorrect gradient detection)
"""

import math

import pytest

from neuralearn.value import Value
from neuralearn.gradient_check import numerical_grad, gradient_check


# ---------------------------------------------------------------------------
# 1. numerical_grad helper
# ---------------------------------------------------------------------------

class TestNumericalGrad:
    """Test the numerical_grad utility function."""

    def test_addition(self):
        a, b = Value(2.0), Value(3.0)
        result = numerical_grad(lambda: (a + b).data, a)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_multiplication(self):
        a, b = Value(2.0), Value(3.0)
        result = numerical_grad(lambda: (a * b).data, a)
        assert result == pytest.approx(3.0, abs=1e-5)

    def test_exp(self):
        a = Value(1.0)
        result = numerical_grad(lambda: math.exp(a.data), a)
        assert result == pytest.approx(math.e, abs=1e-5)

    def test_perturbation_restores_value(self):
        a = Value(5.0)
        original = a.data
        numerical_grad(lambda: a.data ** 2, a)
        assert a.data == pytest.approx(original)


# ---------------------------------------------------------------------------
# 2. gradient_check utility
# ---------------------------------------------------------------------------

class TestGradientCheck:
    """Test the gradient_check utility function."""

    def test_single_input(self):
        a = Value(2.0)
        out = a * 3.0
        out.backward()
        results = gradient_check(lambda: a.data * 3.0, [a])
        assert len(results) == 1
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(3.0, abs=1e-5)
        assert results[0]['numerical'] == pytest.approx(3.0, abs=1e-5)

    def test_multi_input(self):
        a, b = Value(2.0), Value(3.0)
        out = a * b
        out.backward()
        results = gradient_check(lambda: a.data * b.data, [a, b])
        assert len(results) == 2
        assert all(r['passed'] for r in results)
        assert results[0]['analytical'] == pytest.approx(3.0, abs=1e-5)
        assert results[1]['analytical'] == pytest.approx(2.0, abs=1e-5)

    def test_detects_incorrect_gradient(self):
        """The gradient checker should detect a wrong analytical gradient."""
        a = Value(2.0)
        # Correct gradient of a*3 is 3.0, but we set it to 99.0
        a.grad = 99.0
        results = gradient_check(lambda: a.data * 3.0, [a], atol=1e-5, rtol=1e-3)
        assert results[0]['passed'] is False

    def test_passes_results_structure(self):
        a = Value(1.0)
        out = a.exp()
        out.backward()
        results = gradient_check(lambda: math.exp(a.data), [a])
        r = results[0]
        assert 'value' in r
        assert 'analytical' in r
        assert 'numerical' in r
        assert 'abs_error' in r
        assert 'rel_error' in r
        assert 'passed' in r


# ===========================================================================
# 3. Individual Operation Gradient Checks
# ===========================================================================

class TestAdditionGradientCheck:
    def test_both_inputs(self):
        a, b = Value(2.0), Value(3.0)
        out = a + b
        out.backward()
        results = gradient_check(
            lambda: (a.data + b.data), [a, b]
        )
        assert all(r['passed'] for r in results)
        assert results[0]['analytical'] == pytest.approx(1.0)
        assert results[1]['analytical'] == pytest.approx(1.0)


class TestMultiplicationGradientCheck:
    def test_both_inputs(self):
        a, b = Value(2.0), Value(3.0)
        out = a * b
        out.backward()
        results = gradient_check(
            lambda: (a.data * b.data), [a, b]
        )
        assert all(r['passed'] for r in results)
        assert results[0]['analytical'] == pytest.approx(3.0)
        assert results[1]['analytical'] == pytest.approx(2.0)


class TestSubtractionGradientCheck:
    def test_both_inputs(self):
        a, b = Value(5.0), Value(3.0)
        out = a - b
        out.backward()
        results = gradient_check(
            lambda: (a.data - b.data), [a, b]
        )
        assert all(r['passed'] for r in results)
        assert results[0]['analytical'] == pytest.approx(1.0)
        assert results[1]['analytical'] == pytest.approx(-1.0)


class TestPowerGradientCheck:
    def test_square(self):
        a = Value(3.0)
        out = a ** 2
        out.backward()
        results = gradient_check(
            lambda: a.data ** 2, [a]
        )
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(6.0)

    def test_fractional(self):
        a = Value(4.0)
        out = a ** 0.5
        out.backward()
        results = gradient_check(
            lambda: a.data ** 0.5, [a]
        )
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(0.25)


class TestDivisionGradientCheck:
    def test_both_inputs(self):
        a, b = Value(6.0), Value(3.0)
        out = a / b
        out.backward()
        results = gradient_check(
            lambda: (a.data / b.data), [a, b]
        )
        assert all(r['passed'] for r in results)
        assert results[0]['analytical'] == pytest.approx(1.0 / 3.0)
        assert results[1]['analytical'] == pytest.approx(-6.0 / 9.0)

    def test_reflected(self):
        a = Value(3.0)
        out = 6.0 / a
        out.backward()
        results = gradient_check(
            lambda: 6.0 / a.data, [a]
        )
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(-6.0 / 9.0)


class TestReciprocalGradientCheck:
    def test_reciprocal(self):
        a = Value(4.0)
        out = a.reciprocal()
        out.backward()
        results = gradient_check(
            lambda: 1.0 / a.data, [a]
        )
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(-1.0 / 16.0)

    def test_at_one(self):
        a = Value(1.0)
        out = a.reciprocal()
        out.backward()
        results = gradient_check(
            lambda: 1.0 / a.data, [a]
        )
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(-1.0)


class TestExpGradientCheck:
    def test_exp(self):
        a = Value(2.0)
        out = a.exp()
        out.backward()
        results = gradient_check(
            lambda: math.exp(a.data), [a]
        )
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(math.exp(2.0))

    def test_exp_in_chain(self):
        a, b = Value(1.0), Value(2.0)
        out = (a * b).exp()
        out.backward()
        results = gradient_check(
            lambda: math.exp(a.data * b.data), [a, b]
        )
        assert all(r['passed'] for r in results)
        assert results[0]['analytical'] == pytest.approx(math.exp(2.0) * 2.0)
        assert results[1]['analytical'] == pytest.approx(math.exp(2.0) * 1.0)


class TestLogGradientCheck:
    def test_log(self):
        a = Value(5.0)
        out = a.log()
        out.backward()
        results = gradient_check(
            lambda: math.log(a.data), [a]
        )
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(0.2)

    def test_log_in_chain(self):
        a, b = Value(2.0), Value(3.0)
        out = (a * b).log()
        out.backward()
        results = gradient_check(
            lambda: math.log(a.data * b.data), [a, b]
        )
        assert all(r['passed'] for r in results)
        assert results[0]['analytical'] == pytest.approx(0.5)
        assert results[1]['analytical'] == pytest.approx(1.0 / 3.0)


class TestTanhGradientCheck:
    def test_tanh(self):
        a = Value(0.5)
        out = a.tanh()
        out.backward()
        results = gradient_check(
            lambda: math.tanh(a.data), [a]
        )
        assert results[0]['passed'] is True
        expected = 1.0 - math.tanh(0.5) ** 2
        assert results[0]['analytical'] == pytest.approx(expected)

    def test_tanh_in_chain(self):
        a, b = Value(1.0), Value(2.0)
        out = (a + b).tanh()
        out.backward()
        results = gradient_check(
            lambda: math.tanh(a.data + b.data), [a, b]
        )
        assert all(r['passed'] for r in results)
        expected = 1.0 - math.tanh(3.0) ** 2
        assert results[0]['analytical'] == pytest.approx(expected)
        assert results[1]['analytical'] == pytest.approx(expected)


class TestReLUGradientCheck:
    def test_positive(self):
        a = Value(5.0)
        out = a.relu()
        out.backward()
        results = gradient_check(
            lambda: max(0.0, a.data), [a]
        )
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(1.0)

    def test_negative(self):
        a = Value(-3.0)
        out = a.relu()
        out.backward()
        results = gradient_check(
            lambda: max(0.0, a.data), [a]
        )
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(0.0)

    def test_zero_convention(self):
        """ReLU at x=0: gradient is 0 by convention. Numerical gradient
        is approximately 0.5 (the subgradient midpoint), but our convention
        chooses 0. We document this discrepancy rather than asserting a
        numerical gradient match at this nondifferentiable point."""
        a = Value(0.0)
        out = a.relu()
        out.backward()
        # The analytical gradient is 0 (by convention).
        # The numerical gradient is approximately 0.5 (subgradient midpoint).
        # These do NOT match — this is the documented limitation.
        # We verify the convention is applied correctly.
        assert a.grad == pytest.approx(0.0)
        # We do NOT assert gradient_check passes at x=0 because
        # the function is nondifferentiable there.


# ===========================================================================
# 4. Composed Graph Gradient Checks
# ===========================================================================

class TestComposedGraphGradientCheck:
    def test_exp_times_log(self):
        a, b = Value(2.0), Value(3.0)
        out = a.exp() * b.log()
        out.backward()
        results = gradient_check(
            lambda: math.exp(a.data) * math.log(b.data), [a, b]
        )
        assert all(r['passed'] for r in results)
        assert results[0]['analytical'] == pytest.approx(math.exp(2.0) * math.log(3.0))
        assert results[1]['analytical'] == pytest.approx(math.exp(2.0) / 3.0)

    def test_tanh_over_exp(self):
        a = Value(1.0)
        out = a.tanh() / a.exp()
        out.backward()
        sech2 = 1.0 - math.tanh(1.0) ** 2
        expected = (sech2 - math.tanh(1.0)) / math.exp(1.0)
        results = gradient_check(
            lambda: math.tanh(a.data) / math.exp(a.data), [a]
        )
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(expected)

    def test_relu_squared(self):
        a = Value(3.0)
        out = a.relu() ** 2
        out.backward()
        results = gradient_check(
            lambda: max(0.0, a.data) ** 2, [a]
        )
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(6.0)

    def test_log_division(self):
        a, b = Value(10.0), Value(2.0)
        out = a.log() / b
        out.backward()
        results = gradient_check(
            lambda: math.log(a.data) / b.data, [a, b]
        )
        assert all(r['passed'] for r in results)
        assert results[0]['analytical'] == pytest.approx(1.0 / 20.0)
        assert results[1]['analytical'] == pytest.approx(-math.log(10.0) / 4.0)


# ===========================================================================
# 5. Multi-Input Gradient Checks (3+ inputs)
# ===========================================================================

class TestMultiInputGradientCheck:
    def test_three_inputs(self):
        """f(x, y, z) = ((x * y) + exp(z)) / y"""
        x, y, z = Value(2.0), Value(3.0), Value(1.0)
        out = ((x * y) + z.exp()) / y
        out.backward()
        # Numerical verification for all three inputs
        def fn():
            return ((x.data * y.data) + math.exp(z.data)) / y.data
        results = gradient_check(fn, [x, y, z])
        assert all(r['passed'] for r in results)

        # Verify analytical gradients are correct
        # f = (x*y + e^z) / y
        # df/dx = y / y = 1
        assert x.grad == pytest.approx(1.0, abs=1e-4)
        # df/dy = [x*y - (x*y + e^z)] / y^2 = -e^z / y^2 + x/y - x/y = ...
        # Let's compute: f = x + e^z/y
        # df/dy = -e^z/y^2
        assert y.grad == pytest.approx(-math.exp(1.0) / 9.0, abs=1e-4)
        # df/dz = e^z / y
        assert z.grad == pytest.approx(math.exp(1.0) / 3.0, abs=1e-4)

    def test_four_inputs(self):
        """f(a, b, c, d) = a * b + c / d - a * d"""
        a, b, c, d = Value(1.0), Value(2.0), Value(4.0), Value(2.0)
        out = a * b + c / d - a * d
        out.backward()
        def fn():
            return a.data * b.data + c.data / d.data - a.data * d.data
        results = gradient_check(fn, [a, b, c, d])
        assert all(r['passed'] for r in results)
        # df/da = b - d = 0
        assert a.grad == pytest.approx(0.0, abs=1e-4)
        # df/db = a = 1
        assert b.grad == pytest.approx(1.0, abs=1e-4)
        # df/dc = 1/d = 0.5
        assert c.grad == pytest.approx(0.5, abs=1e-4)
        # df/dd = -c/d^2 - a = -1 - 1 = -2
        assert d.grad == pytest.approx(-2.0, abs=1e-4)


# ===========================================================================
# 6. Deep Graph Gradient Checks
# ===========================================================================

class TestDeepGraphGradientCheck:
    def test_deep_chain(self):
        """Chain of 5 multiplications: x * 2 * 2 * 2 * 2 * 2 = 32x"""
        x = Value(1.0)
        a = x * 2.0
        b = a * 2.0
        c = b * 2.0
        d = c * 2.0
        e = d * 2.0
        e.backward()
        # e = 32x, de/dx = 32
        results = gradient_check(
            lambda: (((((x.data * 2.0) * 2.0) * 2.0) * 2.0) * 2.0),
            [x]
        )
        assert results[0]['passed'] is True
        assert results[0]['analytical'] == pytest.approx(32.0)

    def test_deep_with_mixed_ops(self):
        """Deep graph mixing addition, multiplication, division."""
        x, y = Value(2.0), Value(3.0)
        # Build a deep graph
        a = x + y          # 5
        b = a * x          # 10
        c = b / y          # 10/3
        d = c + x          # 10/3 + 2 = 16/3
        e = d * y          # 16
        e.backward()
        def fn():
            a = x.data + y.data
            b = a * x.data
            c = b / y.data
            d = c + x.data
            return d * y.data
        results = gradient_check(fn, [x, y])
        assert all(r['passed'] for r in results)

    def test_deep_stage2_ops(self):
        """Deep graph with exp, log, tanh, division."""
        x = Value(1.0)
        a = x.exp()           # e
        b = a.tanh()          # tanh(e)
        c = b / Value(2.0)    # tanh(e)/2
        d = c.log()           # log(tanh(e)/2)
        d.backward()
        def fn():
            a = math.exp(x.data)
            b = math.tanh(a)
            c = b / 2.0
            return math.log(c)
        results = gradient_check(fn, [x])
        assert results[0]['passed'] is True


# ===========================================================================
# 7. Numerical Edge Cases and Limitations
# ===========================================================================

class TestNumericalEdgeCases:
    def test_near_zero_division(self):
        """Division near zero: numerical gradient is large but finite."""
        a, b = Value(6.0), Value(0.001)
        out = a / b
        out.backward()
        results = gradient_check(
            lambda: (a.data / b.data), [a, b]
        )
        # The gradients are large but the check should still pass
        assert all(r['passed'] for r in results)

    def test_large_exp(self):
        """Large exp values: gradients are large."""
        a = Value(5.0)
        out = a.exp()
        out.backward()
        results = gradient_check(
            lambda: math.exp(a.data), [a]
        )
        assert results[0]['passed'] is True

    def test_saturated_tanh(self):
        """Tanh at large input: gradient approaches zero."""
        a = Value(10.0)
        out = a.tanh()
        out.backward()
        results = gradient_check(
            lambda: math.tanh(a.data), [a]
        )
        assert results[0]['passed'] is True
        # Gradient should be near zero
        assert abs(a.grad) < 1e-4

    def test_small_log(self):
        """Log near zero: gradient is large but finite."""
        a = Value(0.001)
        out = a.log()
        out.backward()
        results = gradient_check(
            lambda: math.log(a.data), [a]
        )
        assert results[0]['passed'] is True


# ===========================================================================
# 8. Regression Tests
# ===========================================================================

class TestGradientCheckRegression:
    def test_does_not_corrupt_analytical_gradients(self):
        """gradient_check must not modify analytical gradients."""
        a, b = Value(2.0), Value(3.0)
        out = a * b
        out.backward()
        grad_a_before = a.grad
        grad_b_before = b.grad
        gradient_check(
            lambda: (a.data * b.data), [a, b]
        )
        assert a.grad == pytest.approx(grad_a_before)
        assert b.grad == pytest.approx(grad_b_before)

    def test_detects_wrong_gradient_addition(self):
        """A deliberately wrong gradient should fail the check."""
        a, b = Value(2.0), Value(3.0)
        out = a + b
        out.backward()
        # Artificially corrupt the gradient
        a.grad = 5.0  # wrong, should be 1.0
        results = gradient_check(
            lambda: (a.data + b.data), [a, b]
        )
        assert results[0]['passed'] is False

    def test_detects_wrong_gradient_multiplication(self):
        a, b = Value(2.0), Value(3.0)
        out = a * b
        out.backward()
        # Artificially corrupt
        b.grad = 99.0  # wrong, should be 2.0
        results = gradient_check(
            lambda: (a.data * b.data), [a, b]
        )
        assert results[1]['passed'] is False

    def test_repeated_backward_with_gradient_check(self):
        """Gradient checking does not interfere with D19 repeated backward."""
        x = Value(2.0)
        y = x * x
        z = y * y
        z.backward()
        g1 = x.grad
        z.backward()
        g2 = x.grad
        # D19: leaf accumulates, so g2 = 2 * g1
        assert g2 == pytest.approx(2 * g1)
        # Now run gradient check — should not affect accumulated gradient
        results = gradient_check(
            lambda: (x.data * x.data) * (x.data * x.data), [x]
        )
        # After gradient_check, x.grad should still be g2 (from repeated backward)
        # gradient_check does not call backward, so it doesn't change .grad
        assert x.grad == pytest.approx(g2)
