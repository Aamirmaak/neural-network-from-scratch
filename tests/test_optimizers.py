"""
Tests for optimizers: SGD, MomentumSGD, Adam.

Covers:
    1. SGD: basic updates, multiple params, zero grad, validation
    2. MomentumSGD: first/second updates, state persistence, validation
    3. Adam: updates, bias correction, state persistence, validation
    4. Integration with Linear layers and losses
    5. Deterministic small learning experiment
    6. D19 semantics and zero_grad lifecycle
"""

import math

import pytest

from neuralearn.value import Value
from neuralearn.parameter import Parameter
from neuralearn.layers import Linear
from neuralearn.losses import mse_loss
from neuralearn.optimizers import SGD, MomentumSGD, Adam


# ---------------------------------------------------------------------------
# 1. SGD
# ---------------------------------------------------------------------------

class TestSGD:
    def test_single_parameter(self):
        """theta_new = theta_old - lr * grad."""
        p = Parameter(5.0)
        opt = SGD([p], lr=0.1)
        p.grad = 2.0
        opt.step()
        assert p.data == pytest.approx(4.8)

    def test_zero_gradient(self):
        """No change when gradient is zero."""
        p = Parameter(5.0)
        opt = SGD([p], lr=0.1)
        p.grad = 0.0
        opt.step()
        assert p.data == pytest.approx(5.0)

    def test_multiple_parameters(self):
        """Each parameter updated independently."""
        p1 = Parameter(10.0)
        p2 = Parameter(20.0)
        opt = SGD([p1, p2], lr=0.05)
        p1.grad = 4.0
        p2.grad = -2.0
        opt.step()
        assert p1.data == pytest.approx(9.8)   # 10 - 0.05*4
        assert p2.data == pytest.approx(20.1)   # 20 - 0.05*(-2)

    def test_multiple_steps(self):
        """Accumulated updates across steps."""
        p = Parameter(0.0)
        opt = SGD([p], lr=0.1)
        for _ in range(10):
            p.grad = 1.0
            opt.step()
        assert p.data == pytest.approx(-1.0)  # 0 - 10*0.1*1

    def test_negative_gradient(self):
        """Negative gradient increases parameter value."""
        p = Parameter(1.0)
        opt = SGD([p], lr=0.1)
        p.grad = -3.0
        opt.step()
        assert p.data == pytest.approx(1.3)  # 1 - 0.1*(-3)

    def test_zero_grad_resets(self):
        """zero_grad clears all parameter gradients."""
        p1 = Parameter(1.0)
        p2 = Parameter(2.0)
        opt = SGD([p1, p2], lr=0.1)
        p1.grad = 5.0
        p2.grad = 3.0
        opt.zero_grad()
        assert p1.grad == 0.0
        assert p2.grad == 0.0

    def test_invalid_lr_zero(self):
        with pytest.raises(ValueError):
            SGD([Parameter(1.0)], lr=0.0)

    def test_invalid_lr_negative(self):
        with pytest.raises(ValueError):
            SGD([Parameter(1.0)], lr=-0.01)

    def test_empty_parameters(self):
        """SGD with no parameters should not raise."""
        opt = SGD([], lr=0.01)
        opt.step()
        opt.zero_grad()

    def test_repr(self):
        opt = SGD([Parameter(1.0), Parameter(2.0)], lr=0.05)
        assert "SGD" in repr(opt)
        assert "0.05" in repr(opt)


# ---------------------------------------------------------------------------
# 2. MomentumSGD
# ---------------------------------------------------------------------------

class TestMomentumSGD:
    def test_first_step(self):
        """v = momentum*0 + grad = grad; theta -= lr * v."""
        p = Parameter(5.0)
        opt = MomentumSGD([p], lr=0.1, momentum=0.9)
        p.grad = 2.0
        opt.step()
        # v = 0.9*0 + 2 = 2; theta = 5 - 0.1*2 = 4.8
        assert p.data == pytest.approx(4.8)

    def test_second_step(self):
        """Velocity accumulates across steps."""
        p = Parameter(5.0)
        opt = MomentumSGD([p], lr=0.1, momentum=0.9)
        p.grad = 2.0
        opt.step()  # v=2, theta=4.8
        p.grad = 2.0
        opt.step()  # v=0.9*2+2=3.8, theta=4.8-0.1*3.8=4.42
        assert p.data == pytest.approx(4.42)

    def test_zero_gradient_preserves_velocity(self):
        """Zero gradient doesn't reset velocity."""
        p = Parameter(5.0)
        opt = MomentumSGD([p], lr=0.1, momentum=0.9)
        p.grad = 2.0
        opt.step()  # v=2, theta=4.8
        p.grad = 0.0
        opt.step()  # v=0.9*2+0=1.8, theta=4.8-0.1*1.8=4.62
        assert p.data == pytest.approx(4.62)

    def test_multiple_parameters(self):
        """Each parameter has independent velocity."""
        p1 = Parameter(10.0)
        p2 = Parameter(20.0)
        opt = MomentumSGD([p1, p2], lr=0.1, momentum=0.9)
        p1.grad = 1.0
        p2.grad = -1.0
        opt.step()
        # p1: v=1, theta=10-0.1=9.9
        # p2: v=-1, theta=20+0.1=20.1
        assert p1.data == pytest.approx(9.9)
        assert p2.data == pytest.approx(20.1)

    def test_zero_momentum(self):
        """momentum=0 behaves like vanilla SGD."""
        p = Parameter(5.0)
        opt = MomentumSGD([p], lr=0.1, momentum=0.0)
        p.grad = 2.0
        opt.step()
        assert p.data == pytest.approx(4.8)
        p.grad = 3.0
        opt.step()
        assert p.data == pytest.approx(4.5)  # 4.8 - 0.1*3

    def test_zero_grad_does_not_reset_velocity(self):
        """zero_grad only clears gradients, not velocity state."""
        p = Parameter(5.0)
        opt = MomentumSGD([p], lr=0.1, momentum=0.9)
        p.grad = 2.0
        opt.step()
        velocity_before = opt._velocity[id(p)]
        opt.zero_grad()
        velocity_after = opt._velocity[id(p)]
        assert velocity_before == velocity_after

    def test_invalid_lr(self):
        with pytest.raises(ValueError):
            MomentumSGD([Parameter(1.0)], lr=-0.01)

    def test_invalid_momentum(self):
        with pytest.raises(ValueError):
            MomentumSGD([Parameter(1.0)], momentum=-0.5)

    def test_repr(self):
        opt = MomentumSGD([Parameter(1.0)], lr=0.01, momentum=0.95)
        assert "MomentumSGD" in repr(opt)
        assert "0.95" in repr(opt)


# ---------------------------------------------------------------------------
# 3. Adam
# ---------------------------------------------------------------------------

class TestAdam:
    def test_first_step_bias_correction(self):
        """First step: m_hat = grad/(1-beta1), v_hat = grad^2/(1-beta2)."""
        p = Parameter(5.0)
        opt = Adam([p], lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        p.grad = 1.0
        opt.step()

        # m = 0.1*1 = 0.1, v = 0.001*1 = 0.001
        # m_hat = 0.1 / 0.1 = 1.0
        # v_hat = 0.001 / 0.001 = 1.0
        # theta = 5 - 0.001 * 1.0 / (1 + 1e-8) ~ 4.999
        expected = 5.0 - 0.001 * 1.0 / (math.sqrt(1.0) + 1e-8)
        assert p.data == pytest.approx(expected, rel=1e-5)

    def test_second_step(self):
        """Moments update across steps."""
        p = Parameter(5.0)
        opt = Adam([p], lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

        p.grad = 1.0
        opt.step()
        data_after_1 = p.data

        p.grad = 1.0
        opt.step()

        # After 2 steps with constant grad=1:
        # m2 = 0.9*0.1 + 0.1*1 = 0.19
        # v2 = 0.999*0.001 + 0.001*1 = 0.001999
        # m_hat = 0.19 / (1-0.9^2) = 0.19/0.19 = 1.0
        # v_hat = 0.001999 / (1-0.999^2) = 0.001999/0.001999 = 1.0
        # theta = data_after_1 - lr * 1.0/(1+eps)
        expected = data_after_1 - 0.001 * 1.0 / (math.sqrt(1.0) + 1e-8)
        assert p.data == pytest.approx(expected, rel=1e-5)

    def test_multiple_parameters(self):
        """Each parameter has independent state."""
        p1 = Parameter(10.0)
        p2 = Parameter(20.0)
        opt = Adam([p1, p2], lr=0.001)
        p1.grad = 1.0
        p2.grad = -1.0
        opt.step()
        # Both should change, but by different signs
        assert p1.data < 10.0
        assert p2.data > 20.0

    def test_zero_gradient(self):
        """Zero gradient produces no update (m and v stay ~0)."""
        p = Parameter(5.0)
        opt = Adam([p], lr=0.001)
        p.grad = 0.0
        opt.step()
        # m=0, v=0, m_hat=0, v_hat=0 -> sqrt(0)+eps=eps -> update ~ 0
        assert p.data == pytest.approx(5.0, abs=1e-6)

    def test_state_persistence(self):
        """Timestep increments across steps."""
        p = Parameter(5.0)
        opt = Adam([p], lr=0.001)
        p.grad = 1.0
        opt.step()
        assert opt._t[id(p)] == 1
        opt.step()
        assert opt._t[id(p)] == 2

    def test_new_parameter_gets_fresh_state(self):
        """Parameters added later get fresh state."""
        p1 = Parameter(1.0)
        opt = Adam([p1], lr=0.001)
        p1.grad = 1.0
        opt.step()
        assert opt._t[id(p1)] == 1

        p2 = Parameter(2.0)
        opt.parameters.append(p2)
        p2.grad = 1.0
        opt.step()
        assert opt._t[id(p2)] == 1  # Fresh start

    def test_zero_grad_does_not_reset_state(self):
        """zero_grad clears gradients but not Adam moments."""
        p = Parameter(5.0)
        opt = Adam([p], lr=0.001)
        p.grad = 1.0
        opt.step()
        m_before = opt._m[id(p)]
        t_before = opt._t[id(p)]
        opt.zero_grad()
        assert opt._m[id(p)] == m_before
        assert opt._t[id(p)] == t_before

    def test_invalid_lr(self):
        with pytest.raises(ValueError):
            Adam([Parameter(1.0)], lr=0.0)

    def test_invalid_beta1(self):
        with pytest.raises(ValueError):
            Adam([Parameter(1.0)], beta1=1.0)

    def test_invalid_beta2(self):
        with pytest.raises(ValueError):
            Adam([Parameter(1.0)], beta2=1.0)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            Adam([Parameter(1.0)], epsilon=0.0)

    def test_repr(self):
        opt = Adam([Parameter(1.0)], lr=0.002, beta1=0.8, beta2=0.99)
        r = repr(opt)
        assert "Adam" in r
        assert "0.002" in r
        assert "0.8" in r


# ---------------------------------------------------------------------------
# 4. Integration with Linear layers and losses
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_sgd_linear_mse(self):
        """Linear -> MSE -> backward -> SGD step."""
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)

        x = [Value(1.0), Value(2.0)]
        target = [Value(5.0)]

        # Forward
        y = layer(x)
        loss = mse_loss(y, target)
        loss.backward()

        # Record pre-update values
        pre_weights = [p.data for p in layer.parameters()]

        # Update
        opt.step()

        # Verify parameters changed
        for p, pre in zip(layer.parameters(), pre_weights):
            assert p.data != pre

    def test_sgd_linear_mse_direction(self):
        """Parameters move in correct gradient direction."""
        layer = Linear(2, 1)
        # Set known weights
        layer.neurons[0].weights[0] = Parameter(1.0)
        layer.neurons[0].weights[1] = Parameter(1.0)
        layer.neurons[0].bias = Parameter(0.0)

        opt = SGD(layer.parameters(), lr=0.1)

        x = [Value(1.0), Value(1.0)]
        target = [Value(5.0)]

        y = layer(x)
        loss = mse_loss(y, target)
        loss.backward()

        # y = 1*1 + 1*1 + 0 = 2.0
        # loss = (2-5)^2 = 9.0
        # dloss/dy = 2*(2-5)/1 = -6.0
        # dy/dw0 = 1.0, dy/dw1 = 1.0, dy/db = 1.0
        # dw0 = -6.0*1.0 = -6.0
        pre_w0 = layer.neurons[0].weights[0].data
        pre_b = layer.neurons[0].bias.data

        opt.step()

        # w0 should increase (negative gradient -> positive update via -= lr*grad)
        assert layer.neurons[0].weights[0].data > pre_w0
        assert layer.neurons[0].bias.data > pre_b

    def test_full_lifecycle(self):
        """forward -> loss -> backward -> step -> zero_grad."""
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)

        x = [Value(1.0), Value(2.0)]
        target = [Value(3.0)]

        y = layer(x)
        loss = mse_loss(y, target)
        loss.backward()

        # Verify gradients exist
        grads = [p.grad for p in layer.parameters()]
        assert any(g != 0.0 for g in grads)

        opt.step()
        opt.zero_grad()

        # Verify gradients reset
        for p in layer.parameters():
            assert p.grad == 0.0

    def test_momentum_linear_mse(self):
        """MomentumSGD with Linear layer."""
        layer = Linear(2, 1)
        opt = MomentumSGD(layer.parameters(), lr=0.01, momentum=0.9)

        x = [Value(1.0), Value(2.0)]
        target = [Value(5.0)]

        y = layer(x)
        loss = mse_loss(y, target)
        loss.backward()
        opt.step()

        # Velocity should be stored
        for p in layer.parameters():
            assert id(p) in opt._velocity

    def test_adam_linear_mse(self):
        """Adam with Linear layer."""
        layer = Linear(2, 1)
        opt = Adam(layer.parameters(), lr=0.001)

        x = [Value(1.0), Value(2.0)]
        target = [Value(5.0)]

        y = layer(x)
        loss = mse_loss(y, target)
        loss.backward()
        opt.step()

        # State should be stored
        for p in layer.parameters():
            assert id(p) in opt._m
            assert id(p) in opt._v
            assert id(p) in opt._t


# ---------------------------------------------------------------------------
# 5. Deterministic small learning experiment
# ---------------------------------------------------------------------------

class TestDeterministicLearning:
    def test_sgd_reduces_loss(self):
        """SGD should reduce MSE over several steps."""
        layer = Linear(1, 1)
        # Set known weights
        layer.neurons[0].weights[0] = Parameter(0.5)
        layer.neurons[0].bias = Parameter(0.0)

        opt = SGD(layer.parameters(), lr=0.05)
        x = [Value(2.0)]
        target = [Value(10.0)]

        losses = []
        for _ in range(20):
            y = layer(x)
            loss = mse_loss(y, target)
            losses.append(loss.data)
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Loss should decrease
        assert losses[-1] < losses[0]
        # Loss should be significantly reduced
        assert losses[-1] < losses[0] * 0.5

    def test_adam_reduces_loss(self):
        """Adam should reduce MSE over several steps."""
        layer = Linear(1, 1)
        layer.neurons[0].weights[0] = Parameter(0.5)
        layer.neurons[0].bias = Parameter(0.0)

        opt = Adam(layer.parameters(), lr=0.05)
        x = [Value(2.0)]
        target = [Value(10.0)]

        losses = []
        for _ in range(30):
            y = layer(x)
            loss = mse_loss(y, target)
            losses.append(loss.data)
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert losses[-1] < losses[0]
        assert losses[-1] < losses[0] * 0.5  # Significant reduction

    def test_momentum_reduces_loss(self):
        """MomentumSGD should reduce MSE over several steps."""
        layer = Linear(1, 1)
        layer.neurons[0].weights[0] = Parameter(0.5)
        layer.neurons[0].bias = Parameter(0.0)

        opt = MomentumSGD(layer.parameters(), lr=0.05, momentum=0.9)
        x = [Value(2.0)]
        target = [Value(10.0)]

        losses = []
        for _ in range(20):
            y = layer(x)
            loss = mse_loss(y, target)
            losses.append(loss.data)
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert losses[-1] < losses[0]
        assert losses[-1] < losses[0] * 0.5

    def test_two_layer_reduces_loss(self):
        """Small 2-layer network should reduce loss."""
        layer1 = Linear(1, 4)
        layer2 = Linear(4, 1)

        params = layer1.parameters() + layer2.parameters()
        opt = Adam(params, lr=0.05)

        x = [Value(1.0)]
        target = [Value(5.0)]

        losses = []
        for _ in range(100):
            h = layer1(x)
            # Apply ReLU manually
            h_act = [hi.relu() for hi in h]
            y = layer2(h_act)
            loss = mse_loss(y, target)
            losses.append(loss.data)
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert losses[-1] < losses[0]
        assert losses[-1] < losses[0] * 0.5


# ---------------------------------------------------------------------------
# 6. D19 semantics and identity preservation
# ---------------------------------------------------------------------------

class TestD19AndIdentity:
    def test_parameter_identity_preserved(self):
        """Optimizer does not replace Parameter objects."""
        p = Parameter(5.0)
        opt = SGD([p], lr=0.1)
        p_ref = p
        p.grad = 1.0
        opt.step()
        assert p is p_ref
        assert isinstance(p, Parameter)

    def test_repeated_backward_accumulates(self):
        """D19: leaf gradients accumulate across backward calls."""
        p = Parameter(2.0)
        opt = SGD([p], lr=0.1)

        # First backward
        loss1 = p * 3.0
        loss1.backward()
        g1 = p.grad

        # Second backward (accumulates)
        loss2 = p * 5.0
        loss2.backward()
        g2 = p.grad

        # g2 should be g1 + new gradient
        assert g2 > g1

    def test_zero_grad_then_backward(self):
        """zero_grad resets, then backward produces fresh gradients."""
        p = Parameter(2.0)
        opt = SGD([p], lr=0.1)

        loss = p * 3.0
        loss.backward()
        assert p.grad != 0.0

        opt.zero_grad()
        assert p.grad == 0.0

        loss2 = p * 5.0
        loss2.backward()
        assert p.grad != 0.0

    def test_step_does_not_affect_grad(self):
        """step() updates data but does not clear gradients."""
        p = Parameter(5.0)
        opt = SGD([p], lr=0.1)
        p.grad = 2.0
        opt.step()
        # Gradient should still be 2.0 after step
        assert p.grad == pytest.approx(2.0)
