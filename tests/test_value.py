"""
Comprehensive tests for the Value class and reverse-mode autodiff (Stages 1 & 2).

Test categories:
    1. Value construction and representation
    2. Basic operations (forward values)
    3. Backward propagation for each operation
    4. Chained expressions
    5. Branching / shared-node gradient accumulation
    6. Scalar interoperability
    7. Numerical gradient checking (finite differences)
    8. Edge cases and regression tests
    9. Stage 2 operations: division, reciprocal, exp, log, tanh, relu
"""

import math

import pytest

from neuralearn.value import Value
from neuralearn.gradient_check import numerical_grad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_close(a, b, tol=1e-6, msg=""):
    """Assert two floats are close, with a useful error message."""
    assert abs(a - b) < tol, (
        f"{msg}Expected {b}, got {a}, diff={abs(a - b)}"
    )


# ===========================================================================
# 1. Value Construction and Representation
# ===========================================================================

class TestValueConstruction:

    def test_stores_data(self):
        v = Value(3.14)
        assert v.data == pytest.approx(3.14)

    def test_gradient_starts_at_zero(self):
        v = Value(5.0)
        assert v.grad == 0.0

    def test_leaf_has_no_parents(self):
        v = Value(1.0)
        assert len(v._prev) == 0

    def test_leaf_has_empty_op(self):
        v = Value(1.0)
        assert v._op == ""

    def test_repr(self):
        v = Value(2.0)
        r = repr(v)
        assert "2.0" in r

    def test_float_conversion(self):
        v = Value(7)
        assert isinstance(v.data, float)
        assert v.data == 7.0


# ===========================================================================
# 2. Basic Operations — Forward Values
# ===========================================================================

class TestAddition:

    def test_value_plus_value(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == pytest.approx(5.0)

    def test_parents_tracked(self):
        a = Value(1.0)
        b = Value(2.0)
        c = a + b
        assert a in c._prev
        assert b in c._prev

    def test_operation_recorded(self):
        a = Value(1.0)
        b = Value(2.0)
        c = a + b
        assert c._op == "+"


class TestMultiplication:

    def test_value_times_value(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == pytest.approx(6.0)

    def test_parents_tracked(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert a in c._prev
        assert b in c._prev

    def test_operation_recorded(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c._op == "*"


class TestNegation:

    def test_neg(self):
        a = Value(3.0)
        b = -a
        assert b.data == pytest.approx(-3.0)

    def test_double_neg(self):
        a = Value(3.0)
        b = -(-a)
        assert b.data == pytest.approx(3.0)


class TestSubtraction:

    def test_value_minus_value(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == pytest.approx(2.0)

    def test_parents_tracked(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert a in c._prev
        assert b in c._prev


class TestPower:

    def test_square(self):
        a = Value(3.0)
        b = a ** 2
        assert b.data == pytest.approx(9.0)

    def test_cube(self):
        a = Value(2.0)
        b = a ** 3
        assert b.data == pytest.approx(8.0)

    def test_power_one(self):
        a = Value(5.0)
        b = a ** 1
        assert b.data == pytest.approx(5.0)

    def test_power_zero(self):
        a = Value(5.0)
        b = a ** 0
        assert b.data == pytest.approx(1.0)

    def test_negative_exponent(self):
        a = Value(2.0)
        b = a ** -1
        assert b.data == pytest.approx(0.5)

    def test_fractional_exponent(self):
        a = Value(4.0)
        b = a ** 0.5
        assert b.data == pytest.approx(2.0)

    def test_type_error_on_non_numeric(self):
        a = Value(2.0)
        with pytest.raises(TypeError):
            a ** "hello"


# ===========================================================================
# 3. Backward Propagation
# ===========================================================================

class TestAdditionBackward:

    def test_simple_add(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        # dc/da = 1, dc/db = 1
        assert a.grad == pytest.approx(1.0)
        assert b.grad == pytest.approx(1.0)

    def test_add_chain(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        d = c + Value(1.0)
        d.backward()
        # dd/da = 1, dd/db = 1
        assert a.grad == pytest.approx(1.0)
        assert b.grad == pytest.approx(1.0)


class TestMultiplicationBackward:

    def test_simple_mul(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        # dc/da = b = 3, dc/db = a = 2
        assert a.grad == pytest.approx(3.0)
        assert b.grad == pytest.approx(2.0)

    def test_mul_chain(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        d = c * Value(2.0)
        d.backward()
        # dd/da = b * 2 = 6, dd/db = a * 2 = 4
        assert a.grad == pytest.approx(6.0)
        assert b.grad == pytest.approx(4.0)


class TestNegationBackward:

    def test_neg(self):
        a = Value(3.0)
        b = -a
        b.backward()
        # db/da = -1
        assert a.grad == pytest.approx(-1.0)


class TestSubtractionBackward:

    def test_simple_sub(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        c.backward()
        # dc/da = 1, dc/db = -1
        assert a.grad == pytest.approx(1.0)
        assert b.grad == pytest.approx(-1.0)


class TestPowerBackward:

    def test_square(self):
        a = Value(3.0)
        c = a ** 2
        c.backward()
        # dc/da = 2*a = 6
        assert a.grad == pytest.approx(6.0)

    def test_cube(self):
        a = Value(2.0)
        c = a ** 3
        c.backward()
        # dc/da = 3*a^2 = 12
        assert a.grad == pytest.approx(12.0)

    def test_negative_exponent(self):
        a = Value(2.0)
        c = a ** -1
        c.backward()
        # dc/da = -1 * a^(-2) = -1/4 = -0.25
        assert a.grad == pytest.approx(-0.25)

    def test_fractional_exponent(self):
        a = Value(4.0)
        c = a ** 0.5
        c.backward()
        # dc/da = 0.5 * 4^(-0.5) = 0.5 * 0.5 = 0.25
        assert a.grad == pytest.approx(0.25)


class TestRootGradient:

    def test_root_gradient_is_one(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        assert c.grad == pytest.approx(1.0)

    def test_root_gradient_independent(self):
        a = Value(1.0)
        b = Value(2.0)
        c = a * b
        c.backward()
        assert c.grad == pytest.approx(1.0)


# ===========================================================================
# 4. Chained Expressions
# ===========================================================================

class TestChainedExpressions:

    def test_z_equals_x_times_y_plus_x(self):
        """z = x*y + x, verify gradients."""
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + x
        z.backward()
        # dz/dy = x = 2
        # dz/dx = y + 1 = 4
        assert z.data == pytest.approx(8.0)
        assert x.grad == pytest.approx(4.0)
        assert y.grad == pytest.approx(2.0)

    def test_z_equals_x_times_y_plus_x_squared(self):
        """z = x*y + x^2, verify gradients."""
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + x ** 2
        z.backward()
        # z = 2*3 + 4 = 10
        # dz/dy = x = 2
        # dz/dx = y + 2x = 3 + 4 = 7
        assert z.data == pytest.approx(10.0)
        assert x.grad == pytest.approx(7.0)
        assert y.grad == pytest.approx(2.0)

    def test_deep_chain(self):
        """z = (a + b) * (a - b), verify gradients."""
        a = Value(3.0)
        b = Value(2.0)
        # (a+b) = 5, (a-b) = 1, z = 5
        z = (a + b) * (a - b)
        z.backward()
        # dz/da = (a-b)*1 + (a+b)*1 = 1 + 5 = 6
        # dz/db = (a-b)*1 + (a+b)*(-1) = 1 - 5 = -4
        assert z.data == pytest.approx(5.0)
        assert a.grad == pytest.approx(6.0)
        assert b.grad == pytest.approx(-4.0)

    def test_nested_power(self):
        """z = (x * y) ** 2"""
        x = Value(2.0)
        y = Value(3.0)
        z = (x * y) ** 2
        z.backward()
        # z = (6)^2 = 36
        # dz/d(x*y) = 2 * (x*y) = 12
        # dz/dx = 12 * y = 36
        # dz/dy = 12 * x = 24
        assert z.data == pytest.approx(36.0)
        assert x.grad == pytest.approx(36.0)
        assert y.grad == pytest.approx(24.0)


# ===========================================================================
# 5. Branching / Shared-Node Gradient Accumulation
# ===========================================================================

class TestBranching:

    def test_x_squared_plus_x(self):
        """z = x*x + x — shared node x appears in two paths."""
        x = Value(3.0)
        z = x * x + x
        z.backward()
        # z = 9 + 3 = 12
        # dz/dx = 2x + 1 = 7
        assert z.data == pytest.approx(12.0)
        assert x.grad == pytest.approx(7.0)

    def test_x_cubed(self):
        """z = x * x * x — shared node x appears in three paths."""
        x = Value(2.0)
        z = x * x * x
        z.backward()
        # z = 8
        # dz/dx = 3 * x^2 = 12
        assert z.data == pytest.approx(8.0)
        assert x.grad == pytest.approx(12.0)

    def test_shared_node_in_add_and_mul(self):
        """x used in both addition and multiplication paths."""
        x = Value(2.0)
        y = Value(3.0)
        # Path 1: x + y -> uses x
        # Path 2: x * y -> uses x
        # z = (x + y) + (x * y) = 5 + 6 = 11
        z = (x + y) + (x * y)
        z.backward()
        # dz/dx from (x+y): 1
        # dz/dx from (x*y): y = 3
        # total dz/dx = 1 + 3 = 4
        # dz/dy from (x+y): 1
        # dz/dy from (x*y): x = 2
        # total dz/dy = 1 + 2 = 3
        assert z.data == pytest.approx(11.0)
        assert x.grad == pytest.approx(4.0)
        assert y.grad == pytest.approx(3.0)

    def test_multi_path_gradient(self):
        """Complex branching: z = (x + x) * (x + x)"""
        x = Value(2.0)
        left = x + x  # 4
        right = x + x  # 4
        z = left * right  # 16
        z.backward()
        # z = (2x)^2 = 4x^2
        # dz/dx = 8x = 16
        assert z.data == pytest.approx(16.0)
        assert x.grad == pytest.approx(16.0)

    def test_four_paths(self):
        """x appears 4 times: z = x + x + x + x"""
        x = Value(2.0)
        z = x + x + x + x
        z.backward()
        # z = 8, dz/dx = 4
        assert z.data == pytest.approx(8.0)
        assert x.grad == pytest.approx(4.0)


# ===========================================================================
# 6. Scalar Interoperability
# ===========================================================================

class TestScalarInteroperability:

    def test_value_plus_scalar(self):
        a = Value(2.0)
        c = a + 3.0
        c.backward()
        assert c.data == pytest.approx(5.0)
        assert a.grad == pytest.approx(1.0)

    def test_scalar_plus_value(self):
        a = Value(2.0)
        c = 3.0 + a
        c.backward()
        assert c.data == pytest.approx(5.0)
        assert a.grad == pytest.approx(1.0)

    def test_value_times_scalar(self):
        a = Value(2.0)
        c = a * 3.0
        c.backward()
        assert c.data == pytest.approx(6.0)
        assert a.grad == pytest.approx(3.0)

    def test_scalar_times_value(self):
        a = Value(2.0)
        c = 3.0 * a
        c.backward()
        assert c.data == pytest.approx(6.0)
        assert a.grad == pytest.approx(3.0)

    def test_value_minus_scalar(self):
        a = Value(5.0)
        c = a - 2.0
        c.backward()
        assert c.data == pytest.approx(3.0)
        assert a.grad == pytest.approx(1.0)

    def test_scalar_minus_value(self):
        a = Value(3.0)
        c = 5.0 - a
        c.backward()
        assert c.data == pytest.approx(2.0)
        assert a.grad == pytest.approx(-1.0)

    def test_value_power_scalar(self):
        a = Value(3.0)
        c = a ** 2
        c.backward()
        assert c.data == pytest.approx(9.0)
        assert a.grad == pytest.approx(6.0)

    def test_mixed_scalar_and_value(self):
        a = Value(2.0)
        c = a * 3 + 1
        c.backward()
        assert c.data == pytest.approx(7.0)
        assert a.grad == pytest.approx(3.0)

    def test_integer_scalar(self):
        a = Value(2.0)
        c = a + 1
        c.backward()
        assert c.data == pytest.approx(3.0)
        assert a.grad == pytest.approx(1.0)

    def test_negative_scalar(self):
        a = Value(2.0)
        c = a + (-1)
        c.backward()
        assert c.data == pytest.approx(1.0)
        assert a.grad == pytest.approx(1.0)


# ===========================================================================
# 7. Numerical Gradient Checking
# ===========================================================================

class TestNumericalGradientCheck:
    """Numerical gradient checking via finite differences.

    Each test creates Value objects outside the function closure,
    so that perturbing their .data actually affects the computation.
    """

    def test_addition(self):
        """z = a + b"""
        a = Value(2.0)
        b = Value(3.0)
        def fn():
            return (a + b).data
        z = a + b
        z.backward()
        num_a = numerical_grad(fn, a)
        assert_close(a.grad, num_a, tol=1e-5, msg="addition w.r.t. a: ")

    def test_multiplication(self):
        """z = a * b"""
        a = Value(2.0)
        b = Value(3.0)
        def fn():
            return (a * b).data
        z = a * b
        z.backward()
        num_a = numerical_grad(fn, a)
        num_b = numerical_grad(fn, b)
        assert_close(a.grad, num_a, tol=1e-5, msg="mul w.r.t. a: ")
        assert_close(b.grad, num_b, tol=1e-5, msg="mul w.r.t. b: ")

    def test_negation(self):
        """z = -a"""
        a = Value(3.0)
        def fn():
            return (-a).data
        z = -a
        z.backward()
        num_a = numerical_grad(fn, a)
        assert_close(a.grad, num_a, tol=1e-5, msg="neg w.r.t. a: ")

    def test_subtraction(self):
        """z = a - b"""
        a = Value(5.0)
        b = Value(3.0)
        def fn():
            return (a - b).data
        z = a - b
        z.backward()
        num_a = numerical_grad(fn, a)
        num_b = numerical_grad(fn, b)
        assert_close(a.grad, num_a, tol=1e-5, msg="sub w.r.t. a: ")
        assert_close(b.grad, num_b, tol=1e-5, msg="sub w.r.t. b: ")

    def test_power(self):
        """z = a ** 2"""
        a = Value(3.0)
        def fn():
            return (a ** 2).data
        z = a ** 2
        z.backward()
        num_a = numerical_grad(fn, a)
        assert_close(a.grad, num_a, tol=1e-5, msg="pow w.r.t. a: ")

    def test_power_negative_exponent(self):
        """z = a ** -1"""
        a = Value(2.0)
        def fn():
            return (a ** -1).data
        z = a ** -1
        z.backward()
        num_a = numerical_grad(fn, a)
        assert_close(a.grad, num_a, tol=1e-5, msg="pow(-1) w.r.t. a: ")

    def test_power_fractional(self):
        """z = a ** 0.5"""
        a = Value(4.0)
        def fn():
            return (a ** 0.5).data
        z = a ** 0.5
        z.backward()
        num_a = numerical_grad(fn, a)
        assert_close(a.grad, num_a, tol=1e-5, msg="pow(0.5) w.r.t. a: ")

    def test_chained_expression(self):
        """z = (a * b + a) ** 2"""
        a = Value(2.0)
        b = Value(3.0)
        def fn():
            return ((a * b + a) ** 2).data
        z = (a * b + a) ** 2
        z.backward()
        num_a = numerical_grad(fn, a)
        num_b = numerical_grad(fn, b)
        assert_close(a.grad, num_a, tol=1e-4, msg="chained w.r.t. a: ")
        assert_close(b.grad, num_b, tol=1e-4, msg="chained w.r.t. b: ")

    def test_branching_expression(self):
        """z = a*a + a*a (4 paths to a)"""
        a = Value(2.0)
        def fn():
            return (a * a + a * a).data
        z = a * a + a * a
        z.backward()
        num_a = numerical_grad(fn, a)
        assert_close(a.grad, num_a, tol=1e-4, msg="branching w.r.t. a: ")

    def test_deep_expression(self):
        """z = ((a + b) * (a - b)) ** 2"""
        a = Value(3.0)
        b = Value(2.0)
        def fn():
            return (((a + b) * (a - b)) ** 2).data
        z = ((a + b) * (a - b)) ** 2
        z.backward()
        num_a = numerical_grad(fn, a)
        num_b = numerical_grad(fn, b)
        assert_close(a.grad, num_a, tol=1e-4, msg="deep w.r.t. a: ")
        assert_close(b.grad, num_b, tol=1e-4, msg="deep w.r.t. b: ")

    def test_many_operations(self):
        """z = a * b + a * 2 + b ** 2"""
        a = Value(2.0)
        b = Value(3.0)
        def fn():
            return (a * b + a * 2 + b ** 2).data
        z = a * b + a * 2 + b ** 2
        z.backward()
        num_a = numerical_grad(fn, a)
        num_b = numerical_grad(fn, b)
        assert_close(a.grad, num_a, tol=1e-4, msg="many_ops w.r.t. a: ")
        assert_close(b.grad, num_b, tol=1e-4, msg="many_ops w.r.t. b: ")


# ===========================================================================
# 8. Edge Cases and Regression
# ===========================================================================

class TestEdgeCases:

    def test_zero_value(self):
        a = Value(0.0)
        b = a * 5
        b.backward()
        assert b.data == pytest.approx(0.0)
        assert a.grad == pytest.approx(5.0)

    def test_negative_values(self):
        a = Value(-2.0)
        b = Value(-3.0)
        c = a * b
        c.backward()
        assert c.data == pytest.approx(6.0)
        assert a.grad == pytest.approx(-3.0)
        assert b.grad == pytest.approx(-2.0)

    def test_large_values(self):
        a = Value(1e6)
        b = Value(2e6)
        c = a * b
        c.backward()
        assert c.data == pytest.approx(2e12)
        assert a.grad == pytest.approx(2e6)
        assert b.grad == pytest.approx(1e6)

    def test_small_values(self):
        a = Value(1e-8)
        b = Value(2e-8)
        c = a * b
        c.backward()
        assert c.data == pytest.approx(2e-16)
        assert a.grad == pytest.approx(2e-8)
        assert b.grad == pytest.approx(1e-8)

    def test_power_zero_exponent(self):
        a = Value(5.0)
        b = a ** 0
        b.backward()
        assert b.data == pytest.approx(1.0)
        assert a.grad == pytest.approx(0.0)

    def test_repeated_backward_calls(self):
        """Calling backward twice should accumulate gradients (leaf-only graph)."""
        a = Value(2.0)
        b = a * 3
        b.backward()
        grad_first = a.grad
        b.backward()
        # After two calls, grad should be doubled
        assert a.grad == pytest.approx(grad_first * 2)

    def test_repeated_backward_with_intermediate_node(self):
        """Repeated backward on graph with depth > 1 and an intermediate node.

        Graph: z = y * y where y = x * x
        z = (x^2)^2 = x^4, dz/dx = 4x^3 = 32 at x=2.

        The intermediate node y must have its gradient freshly computed on each
        backward pass. If stale y.grad is reused, x.grad would be 96 instead
        of the correct 64 after two calls.
        """
        x = Value(2.0)
        y = x * x
        z = y * y

        z.backward()
        assert x.grad == pytest.approx(32.0)
        assert y.grad == pytest.approx(8.0)

        z.backward()
        # Leaf x accumulates: 32 + 32 = 64
        assert x.grad == pytest.approx(64.0)
        # Intermediate y is freshly computed each time (not accumulated)
        assert y.grad == pytest.approx(8.0)

    def test_repeated_backward_branching_graph(self):
        """Repeated backward on a branching/shared-node graph.

        Graph: z = (x + x) * (x + x) = (2x)^2 = 4x^2
        dz/dx = 8x = 16 at x=2.

        Both left and right are intermediate nodes that share leaf x.
        """
        x = Value(2.0)
        left = x + x   # intermediate
        right = x + x  # intermediate
        z = left * right

        z.backward()
        assert x.grad == pytest.approx(16.0)

        z.backward()
        assert x.grad == pytest.approx(32.0)

    def test_repeated_backward_three_calls(self):
        """Three backward calls on a graph with intermediate nodes."""
        x = Value(1.0)
        y = x * 2.0    # intermediate: y = 2
        z = y * 3.0    # root: z = 6, dz/dx = 6

        z.backward()
        assert x.grad == pytest.approx(6.0)

        z.backward()
        assert x.grad == pytest.approx(12.0)

        z.backward()
        assert x.grad == pytest.approx(18.0)

    def test_deep_graph_no_stack_overflow(self):
        """Chain 100 additions — should not overflow."""
        a = Value(1.0)
        for _ in range(100):
            a = a + 1.0
        a.backward()
        assert a.data == pytest.approx(101.0)
        assert a.grad == pytest.approx(1.0)

    def test_deep_mul_chain(self):
        """Chain 5 multiplications by 2: verify original leaf gets correct gradient."""
        x = Value(1.0)
        a = x * 2.0
        b = a * 2.0
        c = b * 2.0
        d = c * 2.0
        e = d * 2.0
        e.backward()
        # e = 32, de/dx = 2^5 = 32
        assert e.data == pytest.approx(32.0)
        assert e.grad == pytest.approx(1.0)  # output grad is always 1
        assert x.grad == pytest.approx(32.0)  # original leaf gradient


# ===========================================================================
# 9. Topological Ordering
# ===========================================================================

class TestTopologicalOrder:

    def test_graph_structure(self):
        """Verify _prev relationships form a correct graph."""
        a = Value(1.0)
        b = Value(2.0)
        c = a + b
        d = a * b
        e = c + d

        assert a in c._prev
        assert b in c._prev
        assert a in d._prev
        assert b in d._prev
        assert c in e._prev
        assert d in e._prev

    def test_leaf_nodes_have_no_parents(self):
        a = Value(1.0)
        b = Value(2.0)
        c = a + b
        assert len(a._prev) == 0
        assert len(b._prev) == 0
        assert len(c._prev) == 2


# ===========================================================================
# 10. Integration-style Tests
# ===========================================================================

class TestIntegration:

    def test_linear_function(self):
        """y = 3*x + 2, verify dy/dx = 3"""
        x = Value(4.0)
        y = x * 3 + 2
        y.backward()
        assert y.data == pytest.approx(14.0)
        assert x.grad == pytest.approx(3.0)

    def test_quadratic_function(self):
        """y = x^2 + 2x + 1 = (x+1)^2, verify dy/dx = 2x + 2"""
        x = Value(3.0)
        y = x ** 2 + x * 2 + 1
        y.backward()
        assert y.data == pytest.approx(16.0)
        assert x.grad == pytest.approx(8.0)

    def test_two_variable_expression(self):
        """f(a,b) = a^2 * b + a * b^2"""
        a = Value(2.0)
        b = Value(3.0)
        f = a ** 2 * b + a * b ** 2
        f.backward()
        # f = 4*3 + 2*9 = 12 + 18 = 30
        # df/da = 2a*b + b^2 = 12 + 9 = 21
        # df/db = a^2 + 2ab = 4 + 12 = 16
        assert f.data == pytest.approx(30.0)
        assert a.grad == pytest.approx(21.0)
        assert b.grad == pytest.approx(16.0)

    def test_expression_with_negation(self):
        """f = -(a * b) + a"""
        a = Value(3.0)
        b = Value(4.0)
        f = -(a * b) + a
        f.backward()
        # f = -12 + 3 = -9
        # df/da = -b + 1 = -3
        # df/db = -a = -3
        assert f.data == pytest.approx(-9.0)
        assert a.grad == pytest.approx(-3.0)
        assert b.grad == pytest.approx(-3.0)

    def test_realistic_neural_computation(self):
        """Simulate a simple neuron: out = sigmoid(w*x + b)
        We test the linear part w*x + b only (no sigmoid in Stage 1)."""
        w = Value(0.5)
        x = Value(2.0)
        b = Value(0.1)
        linear = w * x + b
        linear.backward()
        # linear = 1.1
        # d_linear/dw = x = 2.0
        # d_linear/dx = w = 0.5
        # d_linear/db = 1.0
        assert linear.data == pytest.approx(1.1)
        assert w.grad == pytest.approx(2.0)
        assert x.grad == pytest.approx(0.5)
        assert b.grad == pytest.approx(1.0)


# ===========================================================================
# 11. Regression Tests (bugs found during development)
# ===========================================================================

class TestRegression:

    def test_reflected_addition_does_not_mutate(self):
        """Ensure scalar + Value does not modify the scalar."""
        a = Value(2.0)
        result = 3.0 + a
        assert result.data == pytest.approx(5.0)
        assert a.data == pytest.approx(2.0)

    def test_reflected_mul_does_not_mutate(self):
        """Ensure scalar * Value does not modify the scalar."""
        a = Value(2.0)
        result = 3.0 * a
        assert result.data == pytest.approx(6.0)
        assert a.data == pytest.approx(2.0)

    def test_reflected_sub_direction(self):
        """Ensure scalar - Value gives correct sign."""
        a = Value(3.0)
        result = 5.0 - a
        assert result.data == pytest.approx(2.0)
        result.backward()
        # d(5-a)/da = -1
        assert a.grad == pytest.approx(-1.0)

    def test_power_gradient_near_one(self):
        """Power gradient at x=1 should be n (since n*1^(n-1) = n)."""
        for n in [2, 3, 0.5, -1]:
            a = Value(1.0)
            c = a ** n
            c.backward()
            assert_close(a.grad, n, tol=1e-6,
                         msg=f"pow({n}) at x=1: ")


# ===========================================================================
# Stage 2 — Division
# ===========================================================================

class TestDivision:
    """Forward values for division."""

    def test_value_div_value(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert c.data == pytest.approx(2.0)

    def test_parents_tracked(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert a in c._prev
        assert b in c._prev
        assert c._op == "/"

    def test_div_by_scalar(self):
        a = Value(6.0)
        c = a / 3.0
        assert c.data == pytest.approx(2.0)

    def test_scalar_div_value(self):
        a = Value(3.0)
        c = 6.0 / a
        assert c.data == pytest.approx(2.0)

    def test_negative_values(self):
        a = Value(-6.0)
        b = Value(3.0)
        c = a / b
        assert c.data == pytest.approx(-2.0)

    def test_fractional_result(self):
        a = Value(1.0)
        b = Value(3.0)
        c = a / b
        assert c.data == pytest.approx(1.0 / 3.0)


class TestDivisionBackward:
    """Backward propagation for division."""

    def test_simple_div(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        c.backward()
        # dc/da = 1/b = 1/3
        assert_close(a.grad, 1.0 / 3.0, msg="dc/da: ")
        # dc/db = -a/b² = -6/9 = -2/3
        assert_close(b.grad, -6.0 / 9.0, msg="dc/db: ")

    def test_div_chain(self):
        a = Value(12.0)
        b = Value(2.0)
        c = Value(3.0)
        d = (a / b) / c
        d.backward()
        # d = a / (b * c)
        # dd/da = 1/(b*c) = 1/6
        assert_close(a.grad, 1.0 / 6.0, msg="dd/da chain: ")
        # dd/db = -a/(b²*c) = -12/12 = -1
        assert_close(b.grad, -12.0 / 12.0, msg="dd/db chain: ")
        # dd/dc = -a/(b*c²) = -12/18 = -2/3
        assert_close(c.grad, -12.0 / 18.0, msg="dd/dc chain: ")

    def test_reflected_div_backward(self):
        """scalar / Value backward."""
        a = Value(3.0)
        c = 6.0 / a
        c.backward()
        # dc/da = -6/a² = -6/9 = -2/3
        assert_close(a.grad, -6.0 / 9.0, msg="reflected div: ")


# ===========================================================================
# Stage 2 — Reciprocal
# ===========================================================================

class TestReciprocal:
    """Forward values for reciprocal."""

    def test_reciprocal(self):
        a = Value(4.0)
        c = a.reciprocal()
        assert c.data == pytest.approx(0.25)

    def test_reciprocal_of_one(self):
        a = Value(1.0)
        c = a.reciprocal()
        assert c.data == pytest.approx(1.0)

    def test_reciprocal_negative(self):
        a = Value(-2.0)
        c = a.reciprocal()
        assert c.data == pytest.approx(-0.5)


class TestReciprocalBackward:
    """Backward propagation for reciprocal."""

    def test_simple_reciprocal(self):
        a = Value(4.0)
        c = a.reciprocal()
        c.backward()
        # dc/da = -1/a² = -1/16
        assert_close(a.grad, -1.0 / 16.0, msg="reciprocal grad: ")

    def test_reciprocal_at_one(self):
        a = Value(1.0)
        c = a.reciprocal()
        c.backward()
        # dc/da = -1/1² = -1
        assert_close(a.grad, -1.0, msg="reciprocal at 1: ")


# ===========================================================================
# Stage 2 — Exp
# ===========================================================================

class TestExp:
    """Forward values for exp."""

    def test_exp_zero(self):
        a = Value(0.0)
        c = a.exp()
        assert c.data == pytest.approx(1.0)

    def test_exp_one(self):
        a = Value(1.0)
        c = a.exp()
        assert c.data == pytest.approx(math.e)

    def test_exp_negative(self):
        a = Value(-1.0)
        c = a.exp()
        assert c.data == pytest.approx(1.0 / math.e)

    def test_exp_large(self):
        a = Value(10.0)
        c = a.exp()
        assert c.data == pytest.approx(math.exp(10.0))


class TestExpBackward:
    """Backward propagation for exp."""

    def test_simple_exp(self):
        a = Value(2.0)
        c = a.exp()
        c.backward()
        # dc/da = exp(a) = exp(2)
        assert_close(a.grad, math.exp(2.0), msg="exp grad: ")

    def test_exp_in_chain(self):
        a = Value(1.0)
        b = Value(2.0)
        c = (a * b).exp()
        c.backward()
        # c = exp(a*b), dc/da = exp(a*b) * b
        assert_close(a.grad, math.exp(2.0) * 2.0, msg="exp chain da: ")
        assert_close(b.grad, math.exp(2.0) * 1.0, msg="exp chain db: ")


# ===========================================================================
# Stage 2 — Log
# ===========================================================================

class TestLog:
    """Forward values for log."""

    def test_log_one(self):
        a = Value(1.0)
        c = a.log()
        assert c.data == pytest.approx(0.0)

    def test_log_e(self):
        a = Value(math.e)
        c = a.log()
        assert c.data == pytest.approx(1.0)

    def test_log_large(self):
        a = Value(100.0)
        c = a.log()
        assert c.data == pytest.approx(math.log(100.0))

    def test_log_fractional(self):
        a = Value(0.5)
        c = a.log()
        assert c.data == pytest.approx(math.log(0.5))


class TestLogBackward:
    """Backward propagation for log."""

    def test_simple_log(self):
        a = Value(5.0)
        c = a.log()
        c.backward()
        # dc/da = 1/a = 1/5 = 0.2
        assert_close(a.grad, 0.2, msg="log grad: ")

    def test_log_at_one(self):
        a = Value(1.0)
        c = a.log()
        c.backward()
        # dc/da = 1/1 = 1
        assert_close(a.grad, 1.0, msg="log at 1: ")

    def test_log_in_chain(self):
        a = Value(2.0)
        b = Value(3.0)
        c = (a * b).log()
        c.backward()
        # c = log(a*b), dc/da = 1/(a*b) * b = 1/a
        assert_close(a.grad, 1.0 / 2.0, msg="log chain da: ")
        assert_close(b.grad, 1.0 / 3.0, msg="log chain db: ")


# ===========================================================================
# Stage 2 — Tanh
# ===========================================================================

class TestTanh:
    """Forward values for tanh."""

    def test_tanh_zero(self):
        a = Value(0.0)
        c = a.tanh()
        assert c.data == pytest.approx(0.0)

    def test_tanh_positive(self):
        a = Value(1.0)
        c = a.tanh()
        assert c.data == pytest.approx(math.tanh(1.0))

    def test_tanh_negative(self):
        a = Value(-1.0)
        c = a.tanh()
        assert c.data == pytest.approx(math.tanh(-1.0))

    def test_tanh_large(self):
        a = Value(10.0)
        c = a.tanh()
        assert c.data == pytest.approx(1.0, abs=1e-6)


class TestTanhBackward:
    """Backward propagation for tanh."""

    def test_simple_tanh(self):
        a = Value(0.5)
        c = a.tanh()
        c.backward()
        # dc/da = 1 - tanh(a)²
        expected = 1.0 - math.tanh(0.5) ** 2
        assert_close(a.grad, expected, msg="tanh grad: ")

    def test_tanh_at_zero(self):
        a = Value(0.0)
        c = a.tanh()
        c.backward()
        # dc/da = 1 - tanh(0)² = 1 - 0 = 1
        assert_close(a.grad, 1.0, msg="tanh at 0: ")

    def test_tanh_in_chain(self):
        a = Value(1.0)
        b = Value(2.0)
        c = (a + b).tanh()
        c.backward()
        # c = tanh(a+b), dc/da = (1-tanh(a+b)²) * 1
        expected = 1.0 - math.tanh(3.0) ** 2
        assert_close(a.grad, expected, msg="tanh chain da: ")
        assert_close(b.grad, expected, msg="tanh chain db: ")


# ===========================================================================
# Stage 2 — ReLU
# ===========================================================================

class TestReLU:
    """Forward values for ReLU."""

    def test_relu_positive(self):
        a = Value(5.0)
        c = a.relu()
        assert c.data == pytest.approx(5.0)

    def test_relu_negative(self):
        a = Value(-3.0)
        c = a.relu()
        assert c.data == pytest.approx(0.0)

    def test_relu_zero(self):
        a = Value(0.0)
        c = a.relu()
        assert c.data == pytest.approx(0.0)

    def test_relu_small_positive(self):
        a = Value(0.001)
        c = a.relu()
        assert c.data == pytest.approx(0.001)


class TestReLUBackward:
    """Backward propagation for ReLU."""

    def test_relu_positive_grad(self):
        a = Value(5.0)
        c = a.relu()
        c.backward()
        # dc/da = 1 (since a > 0)
        assert_close(a.grad, 1.0, msg="relu positive: ")

    def test_relu_negative_grad(self):
        a = Value(-3.0)
        c = a.relu()
        c.backward()
        # dc/da = 0 (since a < 0)
        assert_close(a.grad, 0.0, msg="relu negative: ")

    def test_relu_zero_grad(self):
        """Convention: gradient is 0 at x = 0."""
        a = Value(0.0)
        c = a.relu()
        c.backward()
        # dc/da = 0 (convention at x=0)
        assert_close(a.grad, 0.0, msg="relu at 0: ")

    def test_relu_in_chain(self):
        a = Value(2.0)
        b = Value(-1.0)
        c = (a + b).relu()  # relu(1.0) = 1.0
        c.backward()
        # dc/da = 1 (since a+b=1 > 0), dc/db = 1
        assert_close(a.grad, 1.0, msg="relu chain da: ")
        assert_close(b.grad, 1.0, msg="relu chain db: ")

    def test_relu_blocks_negative_gradient(self):
        """ReLU kills gradient for negative inputs."""
        a = Value(-2.0)
        b = Value(3.0)
        c = a.relu() + b  # relu(-2) + 3 = 0 + 3 = 3
        c.backward()
        # dc/da = 0 (killed by relu), dc/db = 1
        assert_close(a.grad, 0.0, msg="relu blocks negative: ")
        assert_close(b.grad, 1.0, msg="relu blocks negative db: ")


# ===========================================================================
# Stage 2 — Numerical Gradient Checking
# ===========================================================================

class TestStage2NumericalGradientCheck:
    """Numerical gradient verification for Stage 2 operations."""

    def test_division(self):
        a, b = Value(6.0), Value(3.0)
        c = a / b
        c.backward()
        def fn():
            return a.data / b.data
        assert numerical_grad(fn, a) == pytest.approx(a.grad, abs=1e-5)
        assert numerical_grad(fn, b) == pytest.approx(b.grad, abs=1e-5)

    def test_exp(self):
        a = Value(2.0)
        c = a.exp()
        c.backward()
        def fn():
            return math.exp(a.data)
        assert numerical_grad(fn, a) == pytest.approx(a.grad, abs=1e-5)

    def test_log(self):
        a = Value(5.0)
        c = a.log()
        c.backward()
        def fn():
            return math.log(a.data)
        assert numerical_grad(fn, a) == pytest.approx(a.grad, abs=1e-5)

    def test_tanh(self):
        a = Value(0.5)
        c = a.tanh()
        c.backward()
        def fn():
            return math.tanh(a.data)
        assert numerical_grad(fn, a) == pytest.approx(a.grad, abs=1e-5)

    def test_relu_positive(self):
        a = Value(2.0)
        c = a.relu()
        c.backward()
        def fn():
            return max(0.0, a.data)
        assert numerical_grad(fn, a) == pytest.approx(a.grad, abs=1e-5)

    def test_div_chain(self):
        a, b, c = Value(6.0), Value(2.0), Value(3.0)
        out = (a / b) * c
        out.backward()
        def fn():
            return (a.data / b.data) * c.data
        assert numerical_grad(fn, a) == pytest.approx(a.grad, abs=1e-5)
        assert numerical_grad(fn, b) == pytest.approx(b.grad, abs=1e-5)
        assert numerical_grad(fn, c) == pytest.approx(c.grad, abs=1e-5)

    def test_exp_log_inverse(self):
        """exp(log(x)) = x, gradient should be 1."""
        a = Value(3.0)
        out = a.log().exp()
        out.backward()
        def fn():
            return math.exp(math.log(a.data))
        assert numerical_grad(fn, a) == pytest.approx(a.grad, abs=1e-5)

    def test_tanh_squared_chain(self):
        a = Value(1.0)
        out = a.tanh() ** 2
        out.backward()
        def fn():
            return math.tanh(a.data) ** 2
        assert numerical_grad(fn, a) == pytest.approx(a.grad, abs=1e-5)

    def test_mixed_stage1_stage2(self):
        """Expression mixing Stage 1 and Stage 2 operations."""
        a, b = Value(2.0), Value(3.0)
        out = a.exp() + b.log() + a * b
        out.backward()
        def fn():
            return math.exp(a.data) + math.log(b.data) + a.data * b.data
        assert numerical_grad(fn, a) == pytest.approx(a.grad, abs=1e-5)
        assert numerical_grad(fn, b) == pytest.approx(b.grad, abs=1e-5)


# ===========================================================================
# Stage 2 — Scalar Interoperability
# ===========================================================================

class TestStage2ScalarInteroperability:
    """Scalar interoperability for Stage 2 operations."""

    def test_value_div_scalar(self):
        a = Value(6.0)
        c = a / 3.0
        assert c.data == pytest.approx(2.0)

    def test_scalar_div_value(self):
        a = Value(3.0)
        c = 6.0 / a
        assert c.data == pytest.approx(2.0)

    def test_exp_of_scalar_value(self):
        """exp() is a method — verify it works on Value."""
        a = Value(2.0)
        c = a.exp()
        assert c.data == pytest.approx(math.exp(2.0))

    def test_log_of_scalar_value(self):
        """log() is a method — verify it works on Value."""
        a = Value(5.0)
        c = a.log()
        assert c.data == pytest.approx(math.log(5.0))


# ===========================================================================
# Stage 2 — Chained Expressions with New Operations
# ===========================================================================

class TestStage2ChainedExpressions:
    """Chained expressions mixing Stage 1 and Stage 2 operations."""

    def test_exp_times_log(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a.exp() * b.log()
        c.backward()
        # dc/da = exp(a) * log(b) = e^2 * ln(3)
        assert_close(a.grad, math.exp(2.0) * math.log(3.0), msg="exp*log da: ")
        # dc/db = exp(a) * (1/b) = e^2 / 3
        assert_close(b.grad, math.exp(2.0) / 3.0, msg="exp*log db: ")

    def test_tanh_over_exp(self):
        a = Value(1.0)
        c = a.tanh() / a.exp()
        c.backward()
        # c = tanh(a) / exp(a)
        # dc/da = [sech²(a)*exp(a) - tanh(a)*exp(a)] / exp(a)²
        #       = [sech²(a) - tanh(a)] / exp(a)
        sech2 = 1.0 - math.tanh(1.0) ** 2
        expected = (sech2 - math.tanh(1.0)) / math.exp(1.0)
        assert_close(a.grad, expected, msg="tanh/exp: ")

    def test_relu_squared(self):
        a = Value(3.0)
        c = a.relu() ** 2
        c.backward()
        # relu(3) = 3, 3^2 = 9, dc/da = 2*3*1 = 6
        assert_close(a.grad, 6.0, msg="relu^2: ")

    def test_log_division(self):
        a = Value(10.0)
        b = Value(2.0)
        c = a.log() / b
        c.backward()
        # dc/da = (1/a) / b = 1/(a*b) = 1/20
        assert_close(a.grad, 1.0 / 20.0, msg="log/div da: ")
        # dc/db = -log(a) / b² = -log(10)/4
        assert_close(b.grad, -math.log(10.0) / 4.0, msg="log/div db: ")


# ===========================================================================
# Stage 2 — Repeated Backward (D19) with New Operations
# ===========================================================================

class TestStage2RepeatedBackward:
    """Repeated backward calls with Stage 2 operations."""

    def test_exp_repeated_backward(self):
        a = Value(2.0)
        c = a.exp()
        c.backward()
        first_grad = a.grad
        c.backward()
        # Leaf gradient accumulates: should be 2 * exp(2)
        assert_close(a.grad, 2.0 * math.exp(2.0), msg="exp repeated: ")

    def test_div_repeated_backward(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        c.backward()
        c.backward()
        # Accumulated: 2 * (1/b) for a, 2 * (-a/b²) for b
        assert_close(a.grad, 2.0 / 3.0, msg="div repeated da: ")
        assert_close(b.grad, 2.0 * (-6.0 / 9.0), msg="div repeated db: ")

    def test_tanh_repeated_backward(self):
        a = Value(1.0)
        c = a.tanh()
        c.backward()
        c.backward()
        expected_grad = 2.0 * (1.0 - math.tanh(1.0) ** 2)
        assert_close(a.grad, expected_grad, msg="tanh repeated: ")

    def test_relu_repeated_backward(self):
        a = Value(5.0)
        c = a.relu()
        c.backward()
        c.backward()
        # relu(5) = 5, grad = 1 each time, accumulated = 2
        assert_close(a.grad, 2.0, msg="relu repeated: ")

    def test_log_repeated_backward(self):
        a = Value(3.0)
        c = a.log()
        c.backward()
        c.backward()
        # grad = 2 * (1/3)
        assert_close(a.grad, 2.0 / 3.0, msg="log repeated: ")
