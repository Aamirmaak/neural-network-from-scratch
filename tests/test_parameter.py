"""
Tests for the Parameter class.

Covers:
    1. Construction and initial state
    2. Gradient access and accumulation
    3. zero_grad behavior
    4. Integration with Value computational graph
    5. Arithmetic operations inherited from Value
"""

import pytest

from neuralearn.value import Value
from neuralearn.parameter import Parameter


# ---------------------------------------------------------------------------
# 1. Construction and initial state
# ---------------------------------------------------------------------------

class TestParameterConstruction:
    def test_stores_data(self):
        p = Parameter(3.0)
        assert p.data == 3.0

    def test_gradient_starts_at_zero(self):
        p = Parameter(3.0)
        assert p.grad == 0.0

    def test_requires_grad_default_true(self):
        p = Parameter(3.0)
        assert p.requires_grad is True

    def test_requires_grad_false(self):
        p = Parameter(3.0, requires_grad=False)
        assert p.requires_grad is False

    def test_is_value(self):
        p = Parameter(3.0)
        assert isinstance(p, Value)

    def test_repr(self):
        p = Parameter(3.0)
        r = repr(p)
        assert "Parameter" in r
        assert "3.0" in r


# ---------------------------------------------------------------------------
# 2. Gradient access and accumulation
# ---------------------------------------------------------------------------

class TestParameterGradient:
    def test_backward_populates_grad(self):
        p = Parameter(2.0)
        out = p * 3.0
        out.backward()
        assert p.grad == pytest.approx(3.0)

    def test_repeated_backward_accumulates(self):
        p = Parameter(2.0)
        out = p * 3.0
        out.backward()
        g1 = p.grad
        out.backward()
        g2 = p.grad
        # D19: leaf gradients accumulate
        assert g2 == pytest.approx(2 * g1)

    def test_multiple_paths(self):
        """Parameter used twice: grad should accumulate from both paths."""
        p = Parameter(2.0)
        out = p * p  # p used twice
        out.backward()
        # d(p*p)/dp = 2p = 4.0
        assert p.grad == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 3. zero_grad behavior
# ---------------------------------------------------------------------------

class TestParameterZeroGrad:
    def test_zero_grad_resets(self):
        p = Parameter(2.0)
        out = p * 3.0
        out.backward()
        assert p.grad != 0.0
        p.zero_grad()
        assert p.grad == 0.0

    def test_zero_grad_then_backward(self):
        p = Parameter(2.0)
        out = p * 3.0
        out.backward()
        p.zero_grad()
        assert p.grad == 0.0
        out.backward()
        assert p.grad == pytest.approx(3.0)

    def test_zero_grad_after_accumulation(self):
        p = Parameter(2.0)
        out = p * p
        out.backward()
        assert p.grad == pytest.approx(4.0)
        p.zero_grad()
        assert p.grad == 0.0


# ---------------------------------------------------------------------------
# 4. Integration with Value computational graph
# ---------------------------------------------------------------------------

class TestParameterGraphIntegration:
    def test_parameter_plus_value(self):
        p = Parameter(2.0)
        v = Value(3.0)
        out = p + v
        out.backward()
        assert p.grad == pytest.approx(1.0)
        assert v.grad == pytest.approx(1.0)

    def test_parameter_in_deep_graph(self):
        p = Parameter(2.0)
        a = p * 3.0
        b = a + 1.0
        c = b.relu()
        c.backward()
        assert p.grad == pytest.approx(3.0)

    def test_two_parameters(self):
        a = Parameter(2.0)
        b = Parameter(3.0)
        out = a * b
        out.backward()
        assert a.grad == pytest.approx(3.0)
        assert b.grad == pytest.approx(2.0)
