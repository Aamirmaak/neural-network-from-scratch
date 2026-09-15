"""
Tests for loss functions: mse_loss and binary_cross_entropy.

Covers:
    1. MSE loss value computation
    2. MSE gradient computation and verification
    3. BCE loss value computation
    4. BCE gradient computation and verification
    5. Numerical gradient checking
    6. Composition with Linear layers
    7. D19 semantics (repeated backward, zero_grad)
    8. Edge cases and error handling
"""

import math

import pytest

from neuralearn.value import Value
from neuralearn.parameter import Parameter
from neuralearn.layers import Linear, ReLU
from neuralearn.losses import mse_loss, binary_cross_entropy
from neuralearn.gradient_check import gradient_check


# ---------------------------------------------------------------------------
# 1. MSE loss value computation
# ---------------------------------------------------------------------------

class TestMSEValues:
    def test_single_perfect_prediction(self):
        loss = mse_loss([Value(3.0)], [Value(3.0)])
        assert loss.data == pytest.approx(0.0)

    def test_single_prediction(self):
        loss = mse_loss([Value(1.0)], [Value(2.0)])
        # (1-2)^2 / 1 = 1.0
        assert loss.data == pytest.approx(1.0)

    def test_multiple_predictions(self):
        loss = mse_loss(
            [Value(1.0), Value(2.0), Value(3.0)],
            [Value(1.5), Value(2.5), Value(3.5)],
        )
        # ((1-1.5)^2 + (2-2.5)^2 + (3-3.5)^2) / 3 = (0.25+0.25+0.25)/3 = 0.25
        assert loss.data == pytest.approx(0.25)

    def test_positive_error(self):
        loss = mse_loss([Value(5.0)], [Value(3.0)])
        # (5-3)^2 / 1 = 4.0
        assert loss.data == pytest.approx(4.0)

    def test_negative_error(self):
        loss = mse_loss([Value(1.0)], [Value(3.0)])
        # (1-3)^2 / 1 = 4.0
        assert loss.data == pytest.approx(4.0)

    def test_zero_targets(self):
        loss = mse_loss([Value(3.0), Value(4.0)], [Value(0.0), Value(0.0)])
        # (9 + 16) / 2 = 12.5
        assert loss.data == pytest.approx(12.5)

    def test_negative_targets(self):
        loss = mse_loss([Value(-1.0), Value(1.0)], [Value(-2.0), Value(2.0)])
        # ((-1-(-2))^2 + (1-2)^2) / 2 = (1+1)/2 = 1.0
        assert loss.data == pytest.approx(1.0)

    def test_mixed_positive_negative(self):
        loss = mse_loss(
            [Value(-2.0), Value(0.0), Value(3.0)],
            [Value(-1.0), Value(0.0), Value(3.0)],
        )
        # ((-2+1)^2 + 0 + 0) / 3 = 1/3
        assert loss.data == pytest.approx(1.0 / 3.0)

    def test_exact_match_all(self):
        preds = [Value(1.0), Value(2.0), Value(3.0)]
        targs = [Value(1.0), Value(2.0), Value(3.0)]
        loss = mse_loss(preds, targs)
        assert loss.data == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. MSE gradient computation
# ---------------------------------------------------------------------------

class TestMSEGradients:
    def test_single_gradient(self):
        """dMSE/dy_hat = 2(y_hat - y) / n."""
        pred = Value(1.0)
        loss = mse_loss([pred], [Value(2.0)])
        loss.backward()
        # d/d(1.0) of (1-2)^2/1 = 2*(1-2)/1 = -2.0
        assert pred.grad == pytest.approx(-2.0)

    def test_multiple_gradient(self):
        """Each prediction gets its own gradient."""
        p1, p2, p3 = Value(1.0), Value(2.0), Value(3.0)
        loss = mse_loss([p1, p2, p3], [Value(1.5), Value(2.5), Value(3.5)])
        loss.backward()
        # dp1: 2*(1-1.5)/3 = -1/3
        # dp2: 2*(2-2.5)/3 = -1/3
        # dp3: 2*(3-3.5)/3 = -1/3
        assert p1.grad == pytest.approx(-1.0 / 3.0)
        assert p2.grad == pytest.approx(-1.0 / 3.0)
        assert p3.grad == pytest.approx(-1.0 / 3.0)

    def test_gradient_sign(self):
        """Gradient points in direction that reduces loss."""
        pred = Value(5.0)
        loss = mse_loss([pred], [Value(3.0)])
        loss.backward()
        # pred is too high, gradient should be positive (increase pred increases loss)
        # d/d(5) of (5-3)^2 = 2*(5-3) = 4.0
        assert pred.grad == pytest.approx(4.0)

    def test_gradient_symmetry(self):
        """Errors of equal magnitude produce equal gradient magnitudes."""
        p1, p2 = Value(1.0), Value(5.0)
        loss = mse_loss([p1, p2], [Value(2.0), Value(4.0)])
        loss.backward()
        # dp1: 2*(1-2)/2 = -1.0
        # dp2: 2*(5-4)/2 = 1.0
        assert p1.grad == pytest.approx(-1.0)
        assert p2.grad == pytest.approx(1.0)

    def test_zero_error_gradient(self):
        """Perfect prediction has zero gradient."""
        pred = Value(3.0)
        loss = mse_loss([pred], [Value(3.0)])
        loss.backward()
        assert pred.grad == pytest.approx(0.0)

    def test_repeated_backward_accumulates(self):
        """D19: leaf gradients accumulate across backward calls."""
        pred = Value(1.0)
        loss = mse_loss([pred], [Value(2.0)])
        loss.backward()
        g1 = pred.grad
        loss.backward()
        g2 = pred.grad
        assert g2 == pytest.approx(2 * g1)


# ---------------------------------------------------------------------------
# 3. BCE loss value computation
# ---------------------------------------------------------------------------

class TestBCEValues:
    def test_perfect_positive(self):
        """y=1, p=1 -> BCE = -log(1) = 0."""
        loss = binary_cross_entropy([Value(1.0)], [Value(1.0)])
        assert loss.data == pytest.approx(0.0, abs=1e-6)

    def test_perfect_negative(self):
        """y=0, p=0 -> BCE = -log(1) = 0."""
        loss = binary_cross_entropy([Value(0.0)], [Value(0.0)])
        assert loss.data == pytest.approx(0.0, abs=1e-6)

    def test_positive_target(self):
        """y=1, p=0.9 -> BCE = -log(0.9)."""
        loss = binary_cross_entropy([Value(0.9)], [Value(1.0)])
        assert loss.data == pytest.approx(-math.log(0.9), abs=1e-5)

    def test_negative_target(self):
        """y=0, p=0.1 -> BCE = -log(0.9)."""
        loss = binary_cross_entropy([Value(0.1)], [Value(0.0)])
        assert loss.data == pytest.approx(-math.log(0.9), abs=1e-5)

    def test_multiple_samples(self):
        """Mean of individual BCE terms."""
        preds = [Value(0.9), Value(0.1)]
        targs = [Value(1.0), Value(0.0)]
        loss = binary_cross_entropy(preds, targs)
        expected = -(math.log(0.9) + math.log(0.9)) / 2
        assert loss.data == pytest.approx(expected, abs=1e-5)

    def test_mixed_targets(self):
        """y=1 and y=0 in same batch."""
        preds = [Value(0.8), Value(0.3)]
        targs = [Value(1.0), Value(0.0)]
        loss = binary_cross_entropy(preds, targs)
        expected = -(math.log(0.8) + math.log(0.7)) / 2
        assert loss.data == pytest.approx(expected, abs=1e-5)

    def test_high_confidence_correct(self):
        """y=1, p=0.99 -> BCE is very small."""
        loss = binary_cross_entropy([Value(0.99)], [Value(1.0)])
        assert loss.data == pytest.approx(-math.log(0.99), abs=1e-5)

    def test_high_confidence_wrong(self):
        """y=1, p=0.01 -> BCE is very large."""
        loss = binary_cross_entropy([Value(0.01)], [Value(1.0)])
        assert loss.data == pytest.approx(-math.log(0.01), abs=1e-3)

    def test_clipping_at_zero(self):
        """p=0 is clipped to eps, BCE is finite."""
        loss = binary_cross_entropy([Value(0.0)], [Value(1.0)])
        assert math.isfinite(loss.data)
        assert loss.data > 0

    def test_clipping_at_one(self):
        """p=1 is clipped to 1-eps, BCE is finite."""
        loss = binary_cross_entropy([Value(1.0)], [Value(0.0)])
        assert math.isfinite(loss.data)
        assert loss.data > 0

    def test_negative_prediction_clipped(self):
        """p < 0 is clipped to eps."""
        loss = binary_cross_entropy([Value(-1.0)], [Value(1.0)])
        assert math.isfinite(loss.data)

    def test_prediction_above_one_clipped(self):
        """p > 1 is clipped to 1-eps."""
        loss = binary_cross_entropy([Value(2.0)], [Value(0.0)])
        assert math.isfinite(loss.data)


# ---------------------------------------------------------------------------
# 4. BCE gradient computation
# ---------------------------------------------------------------------------

class TestBCEGradients:
    def test_single_positive_gradient(self):
        """dBCE/dp = -y/p + (1-y)/(1-p). For y=1: dBCE/dp = -1/p."""
        pred = Value(0.8)
        loss = binary_cross_entropy([pred], [Value(1.0)])
        loss.backward()
        # dBCE/dp = -1/0.8 = -1.25
        assert pred.grad == pytest.approx(-1.25, abs=1e-4)

    def test_single_negative_gradient(self):
        """For y=0: dBCE/dp = 1/(1-p)."""
        pred = Value(0.3)
        loss = binary_cross_entropy([pred], [Value(0.0)])
        loss.backward()
        # dBCE/dp = 1/(1-0.3) = 1/0.7 ~ 1.4286
        assert pred.grad == pytest.approx(1.0 / 0.7, abs=1e-4)

    def test_multiple_gradient(self):
        """Multiple predictions get independent gradients."""
        p1, p2 = Value(0.8), Value(0.3)
        loss = binary_cross_entropy([p1, p2], [Value(1.0), Value(0.0)])
        loss.backward()
        # dp1: -1/(2*0.8) = -0.625
        # dp2: 1/(2*0.7) ~ 0.7143
        assert p1.grad == pytest.approx(-1.0 / (2 * 0.8), abs=1e-4)
        assert p2.grad == pytest.approx(1.0 / (2 * 0.7), abs=1e-4)

    def test_gradient_sign(self):
        """Gradient points toward reducing loss."""
        pred = Value(0.3)
        loss = binary_cross_entropy([pred], [Value(1.0)])
        loss.backward()
        # p is too low for y=1, gradient should be negative (increase p decreases loss)
        assert pred.grad < 0

    def test_repeated_backward_accumulates(self):
        """D19: leaf gradients accumulate across backward calls."""
        pred = Value(0.5)
        loss = binary_cross_entropy([pred], [Value(1.0)])
        loss.backward()
        g1 = pred.grad
        loss.backward()
        g2 = pred.grad
        assert g2 == pytest.approx(2 * g1)


# ---------------------------------------------------------------------------
# 5. Numerical gradient checking
# ---------------------------------------------------------------------------

class TestMSEGradientCheck:
    def test_single_prediction(self):
        """Numerical gradient matches analytical for single MSE."""
        pred = Value(1.5)
        loss = mse_loss([pred], [Value(2.0)])
        loss.backward()
        results = gradient_check(
            lambda: mse_loss([Value(pred.data)], [Value(2.0)]).data,
            [pred],
        )
        assert all(r["passed"] for r in results)

    def test_multiple_predictions(self):
        """Numerical gradient matches analytical for multi-sample MSE."""
        p1, p2, p3 = Value(1.0), Value(2.0), Value(3.0)
        loss = mse_loss([p1, p2, p3], [Value(1.5), Value(2.5), Value(3.5)])
        loss.backward()
        results = gradient_check(
            lambda: mse_loss(
                [Value(p1.data), Value(p2.data), Value(p3.data)],
                [Value(1.5), Value(2.5), Value(3.5)],
            ).data,
            [p1, p2, p3],
        )
        assert all(r["passed"] for r in results)


class TestBCEGradientCheck:
    def test_mid_range(self):
        """Numerical gradient matches analytical for p=0.5, y=1."""
        pred = Value(0.5)
        loss = binary_cross_entropy([pred], [Value(1.0)])
        loss.backward()
        results = gradient_check(
            lambda: binary_cross_entropy([Value(pred.data)], [Value(1.0)]).data,
            [pred],
        )
        assert all(r["passed"] for r in results)

    def test_near_zero(self):
        """Numerical gradient for p near 0 (clipped)."""
        pred = Value(0.01)
        loss = binary_cross_entropy([pred], [Value(1.0)])
        loss.backward()
        results = gradient_check(
            lambda: binary_cross_entropy([Value(pred.data)], [Value(1.0)]).data,
            [pred],
            eps=1e-6,
        )
        assert all(r["passed"] for r in results)

    def test_near_one(self):
        """Numerical gradient for p near 1 (clipped)."""
        pred = Value(0.99)
        loss = binary_cross_entropy([pred], [Value(0.0)])
        loss.backward()
        results = gradient_check(
            lambda: binary_cross_entropy([Value(pred.data)], [Value(0.0)]).data,
            [pred],
            eps=1e-6,
        )
        assert all(r["passed"] for r in results)

    def test_target_zero(self):
        """Gradient check with y=0."""
        pred = Value(0.7)
        loss = binary_cross_entropy([pred], [Value(0.0)])
        loss.backward()
        results = gradient_check(
            lambda: binary_cross_entropy([Value(pred.data)], [Value(0.0)]).data,
            [pred],
        )
        assert all(r["passed"] for r in results)

    def test_target_one(self):
        """Gradient check with y=1."""
        pred = Value(0.7)
        loss = binary_cross_entropy([pred], [Value(1.0)])
        loss.backward()
        results = gradient_check(
            lambda: binary_cross_entropy([Value(pred.data)], [Value(1.0)]).data,
            [pred],
        )
        assert all(r["passed"] for r in results)

    def test_multiple_samples(self):
        """Gradient check with multiple samples."""
        p1, p2 = Value(0.6), Value(0.4)
        loss = binary_cross_entropy([p1, p2], [Value(1.0), Value(0.0)])
        loss.backward()
        results = gradient_check(
            lambda: binary_cross_entropy(
                [Value(p1.data), Value(p2.data)],
                [Value(1.0), Value(0.0)],
            ).data,
            [p1, p2],
        )
        assert all(r["passed"] for r in results)


# ---------------------------------------------------------------------------
# 6. Composition with Linear layers
# ---------------------------------------------------------------------------

class TestMSEComposition:
    def test_linear_mse_backward(self):
        """Linear -> MSE -> backward: gradients reach weights and biases."""
        layer = Linear(2, 1)
        x = [Value(1.0), Value(2.0)]
        target = [Value(3.0)]

        y = layer(x)
        loss = mse_loss(y, target)
        loss.backward()

        # Verify all parameters have non-zero gradients
        for p in layer.parameters():
            assert p.grad != 0.0

    def test_linear_mse_gradient_values(self):
        """Verify specific gradient values for Linear -> MSE."""
        layer = Linear(2, 1)
        # Set known weights
        layer.neurons[0].weights[0] = Parameter(1.0)
        layer.neurons[0].weights[1] = Parameter(1.0)
        layer.neurons[0].bias = Parameter(0.0)

        x = [Value(1.0), Value(2.0)]
        target = [Value(5.0)]

        y = layer(x)
        loss = mse_loss(y, target)
        loss.backward()

        # y = 1*1 + 1*2 + 0 = 3.0
        # loss = (3-5)^2 = 4.0
        # dloss/dy = 2*(3-5)/1 = -4.0
        # dy/dw0 = x0 = 1.0, dy/dw1 = x1 = 2.0, dy/db = 1.0
        # dloss/dw0 = -4.0 * 1.0 = -4.0
        # dloss/dw1 = -4.0 * 2.0 = -8.0
        # dloss/db = -4.0 * 1.0 = -4.0
        assert layer.neurons[0].weights[0].grad == pytest.approx(-4.0)
        assert layer.neurons[0].weights[1].grad == pytest.approx(-8.0)
        assert layer.neurons[0].bias.grad == pytest.approx(-4.0)

    def test_linear_mse_zero_grad(self):
        """After backward, zero_grad resets all parameter gradients."""
        layer = Linear(2, 1)
        x = [Value(1.0), Value(2.0)]
        y = layer(x)
        loss = mse_loss(y, [Value(3.0)])
        loss.backward()

        # Gradients should be non-zero
        grads_before = [p.grad for p in layer.parameters()]
        assert any(g != 0.0 for g in grads_before)

        # Zero grad
        layer.zero_grad()
        grads_after = [p.grad for p in layer.parameters()]
        assert all(g == 0.0 for g in grads_after)


class TestBCEComposition:
    def test_linear_bce_backward(self):
        """Linear -> sigmoid -> BCE -> backward: gradients reach weights."""
        layer = Linear(2, 1)
        x = [Value(1.0), Value(2.0)]

        y = layer(x)

        # Apply sigmoid to convert logits to probabilities in (0, 1)
        # sigmoid(x) = 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))
        probs = [1.0 / (1.0 + (-yi).exp()) for yi in y]

        target = [Value(1.0)]
        loss = binary_cross_entropy(probs, target)
        loss.backward()

        # Verify all parameters have non-zero gradients
        for p in layer.parameters():
            assert p.grad != 0.0

    def test_relu_linear_bce(self):
        """ReLU -> Linear -> sigmoid -> BCE -> backward."""
        layer = Linear(2, 1)
        act = ReLU()
        # Set weights so output is positive (ReLU doesn't zero it)
        layer.neurons[0].weights[0] = Parameter(1.0)
        layer.neurons[0].weights[1] = Parameter(1.0)
        layer.neurons[0].bias = Parameter(1.0)
        x = [Value(1.0), Value(1.0)]

        y_raw = layer(x)
        y_act = [act(yi) for yi in y_raw]
        # y = 1*1 + 1*1 + 1 = 3.0, ReLU(3) = 3.0 (positive)

        # Apply sigmoid to convert to probabilities
        probs = [1.0 / (1.0 + (-yi).exp()) for yi in y_act]

        target = [Value(1.0)]
        loss = binary_cross_entropy(probs, target)
        loss.backward()

        # Verify all parameters have non-zero gradients
        for p in layer.parameters():
            assert p.grad != 0.0


# ---------------------------------------------------------------------------
# 7. D19 semantics
# ---------------------------------------------------------------------------

class TestD19Semantics:
    def test_mse_repeated_backward(self):
        """MSE: repeated backward accumulates leaf gradients."""
        pred = Value(2.0)
        loss = mse_loss([pred], [Value(3.0)])
        loss.backward()
        g1 = pred.grad
        loss.backward()
        g2 = pred.grad
        # D19: g2 should be 2*g1
        assert g2 == pytest.approx(2 * g1)

    def test_bce_repeated_backward(self):
        """BCE: repeated backward accumulates leaf gradients."""
        pred = Value(0.5)
        loss = binary_cross_entropy([pred], [Value(1.0)])
        loss.backward()
        g1 = pred.grad
        loss.backward()
        g2 = pred.grad
        assert g2 == pytest.approx(2 * g1)

    def test_linear_mse_zero_grad_cycle(self):
        """Forward -> backward -> zero_grad -> backward cycle."""
        layer = Linear(2, 1)
        x = [Value(1.0), Value(2.0)]

        # First pass
        y = layer(x)
        loss = mse_loss(y, [Value(3.0)])
        loss.backward()
        grads1 = [p.grad for p in layer.parameters()]

        # Zero grad
        layer.zero_grad()
        assert all(p.grad == 0.0 for p in layer.parameters())

        # Second pass
        y = layer(x)
        loss = mse_loss(y, [Value(3.0)])
        loss.backward()
        grads2 = [p.grad for p in layer.parameters()]

        # Gradients should be identical (same graph, fresh accumulation)
        for g1, g2 in zip(grads1, grads2):
            assert g2 == pytest.approx(g1)


# ---------------------------------------------------------------------------
# 8. Error handling
# ---------------------------------------------------------------------------

class TestMSEErrors:
    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            mse_loss([Value(1.0)], [Value(1.0), Value(2.0)])

    def test_empty_input(self):
        with pytest.raises(ValueError):
            mse_loss([], [])


class TestBCEErrors:
    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            binary_cross_entropy([Value(0.5)], [Value(0.5), Value(0.5)])

    def test_empty_input(self):
        with pytest.raises(ValueError):
            binary_cross_entropy([], [])


# ---------------------------------------------------------------------------
# 9. Regression tests
# ---------------------------------------------------------------------------

class TestRegression:
    def test_mse_nonnegative(self):
        """MSE is always non-negative."""
        for p, t in [(0.0, 1.0), (5.0, 3.0), (-1.0, 2.0), (0.0, 0.0)]:
            loss = mse_loss([Value(p)], [Value(t)])
            assert loss.data >= 0.0

    def test_bce_nonnegative(self):
        """BCE is always non-negative for valid probabilities."""
        for p, t in [(0.1, 0.0), (0.9, 1.0), (0.5, 1.0), (0.5, 0.0)]:
            loss = binary_cross_entropy([Value(p)], [Value(t)])
            assert loss.data >= 0.0

    def test_mse_is_value(self):
        """mse_loss returns a Value."""
        loss = mse_loss([Value(1.0)], [Value(2.0)])
        assert isinstance(loss, Value)

    def test_bce_is_value(self):
        """binary_cross_entropy returns a Value."""
        loss = binary_cross_entropy([Value(0.5)], [Value(1.0)])
        assert isinstance(loss, Value)

    def test_mse_graph_connected(self):
        """MSE loss is connected to predictions in the graph."""
        pred = Value(1.0)
        loss = mse_loss([pred], [Value(2.0)])
        loss.backward()
        # pred is a leaf node, so it should have received a gradient
        assert pred.grad != 0.0

    def test_scalar_interop(self):
        """Loss functions accept plain floats as targets."""
        loss = mse_loss([Value(1.0)], [2.0])
        assert loss.data == pytest.approx(1.0)

        loss = binary_cross_entropy([Value(0.5)], [1.0])
        assert math.isfinite(loss.data)
