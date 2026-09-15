"""
Tests for neural-network layers: Module, Neuron, Linear, ReLU, Tanh.

Covers:
    1. Module base class
    2. Neuron construction, forward, parameters, gradients
    3. Linear construction, forward, parameters, gradients
    4. ReLU and Tanh layers
    5. Gradient checking for layers
    6. Edge cases
    7. Regression (existing tests unaffected)
"""

import math

import pytest

from neuralearn.value import Value
from neuralearn.parameter import Parameter
from neuralearn.layers import Module, Neuron, Linear, ReLU, Tanh
from neuralearn.gradient_check import numerical_grad, gradient_check


# ---------------------------------------------------------------------------
# 1. Module base class
# ---------------------------------------------------------------------------

class TestModule:
    def test_forward_not_implemented(self):
        m = Module()
        with pytest.raises(NotImplementedError):
            m.forward()

    def test_parameters_empty(self):
        m = Module()
        assert m.parameters() == []

    def test_zero_grad_empty(self):
        m = Module()
        m.zero_grad()  # should not raise

    def test_callable(self):
        """Module.__call__ delegates to forward."""
        class Double(Module):
            def forward(self, x):
                return x * 2.0
        m = Double()
        out = m(Value(3.0))
        assert out.data == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# 2. Neuron
# ---------------------------------------------------------------------------

class TestNeuronConstruction:
    def test_parameter_count(self):
        n = Neuron(3)
        assert len(n.parameters()) == 4  # 3 weights + 1 bias

    def test_single_input(self):
        n = Neuron(1)
        assert len(n.parameters()) == 2

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            Neuron(0)

    def test_repr(self):
        n = Neuron(3)
        assert "Neuron" in repr(n)
        assert "3" in repr(n)


class TestNeuronForward:
    def test_weighted_sum(self):
        """Manually set weights and verify output."""
        n = Neuron(2)
        # Set known weights
        n.weights[0] = Parameter(1.0)
        n.weights[1] = Parameter(2.0)
        n.bias = Parameter(0.5)
        # y = 1.0*3.0 + 2.0*4.0 + 0.5 = 3 + 8 + 0.5 = 11.5
        x = [Value(3.0), Value(4.0)]
        out = n(x)
        assert out.data == pytest.approx(11.5)

    def test_zero_bias(self):
        n = Neuron(2)
        n.weights[0] = Parameter(1.0)
        n.weights[1] = Parameter(1.0)
        n.bias = Parameter(0.0)
        x = [Value(2.0), Value(3.0)]
        out = n(x)
        assert out.data == pytest.approx(5.0)

    def test_wrong_input_count(self):
        n = Neuron(3)
        with pytest.raises(ValueError):
            n([Value(1.0), Value(2.0)])


class TestNeuronGradients:
    def test_weight_gradients(self):
        """dy/dw_i = x_i."""
        n = Neuron(3)
        n.weights[0] = Parameter(1.0)
        n.weights[1] = Parameter(1.0)
        n.weights[2] = Parameter(1.0)
        n.bias = Parameter(0.0)
        x = [Value(2.0), Value(3.0), Value(4.0)]
        out = n(x)
        out.backward()
        # dy/dw_i = x_i
        assert n.weights[0].grad == pytest.approx(2.0)
        assert n.weights[1].grad == pytest.approx(3.0)
        assert n.weights[2].grad == pytest.approx(4.0)

    def test_bias_gradient(self):
        """dy/db = 1."""
        n = Neuron(2)
        n.weights[0] = Parameter(1.0)
        n.weights[1] = Parameter(1.0)
        n.bias = Parameter(0.0)
        x = [Value(1.0), Value(1.0)]
        out = n(x)
        out.backward()
        assert n.bias.grad == pytest.approx(1.0)

    def test_input_gradients(self):
        """dy/dx_i = w_i."""
        n = Neuron(3)
        n.weights[0] = Parameter(2.0)
        n.weights[1] = Parameter(3.0)
        n.weights[2] = Parameter(4.0)
        n.bias = Parameter(0.0)
        x = [Value(1.0), Value(1.0), Value(1.0)]
        out = n(x)
        out.backward()
        assert x[0].grad == pytest.approx(2.0)
        assert x[1].grad == pytest.approx(3.0)
        assert x[2].grad == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 3. Linear layer
# ---------------------------------------------------------------------------

class TestLinearConstruction:
    def test_parameter_count(self):
        """nout * (nin + 1) parameters."""
        lin = Linear(3, 2)
        assert len(lin.parameters()) == 2 * (3 + 1)  # 8

    def test_single_neuron(self):
        lin = Linear(1, 1)
        assert len(lin.parameters()) == 2

    def test_attributes(self):
        lin = Linear(3, 2)
        assert lin.nin == 3
        assert lin.nout == 2
        assert len(lin.neurons) == 2

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            Linear(0, 2)
        with pytest.raises(ValueError):
            Linear(2, 0)


class TestLinearForward:
    def test_output_count(self):
        lin = Linear(3, 2)
        x = [Value(1.0), Value(2.0), Value(3.0)]
        y = lin(x)
        assert len(y) == 2

    def test_manual_weights(self):
        """Verify output with known weights."""
        lin = Linear(2, 1)
        # Set weights manually
        lin.neurons[0].weights[0] = Parameter(1.0)
        lin.neurons[0].weights[1] = Parameter(2.0)
        lin.neurons[0].bias = Parameter(0.5)
        x = [Value(3.0), Value(4.0)]
        y = lin(x)
        # y = 1*3 + 2*4 + 0.5 = 11.5
        assert y[0].data == pytest.approx(11.5)

    def test_two_neurons_independent(self):
        """Each neuron should have independent weights."""
        lin = Linear(2, 2)
        lin.neurons[0].weights[0] = Parameter(1.0)
        lin.neurons[0].weights[1] = Parameter(0.0)
        lin.neurons[0].bias = Parameter(0.0)
        lin.neurons[1].weights[0] = Parameter(0.0)
        lin.neurons[1].weights[1] = Parameter(1.0)
        lin.neurons[1].bias = Parameter(0.0)
        x = [Value(3.0), Value(4.0)]
        y = lin(x)
        assert y[0].data == pytest.approx(3.0)
        assert y[1].data == pytest.approx(4.0)

    def test_wrong_input_count(self):
        lin = Linear(3, 2)
        with pytest.raises(ValueError):
            lin([Value(1.0), Value(2.0)])


class TestLinearGradients:
    def test_weight_gradient(self):
        """dy_j/dw_{j,i} = x_i for neuron j."""
        lin = Linear(2, 1)
        lin.neurons[0].weights[0] = Parameter(1.0)
        lin.neurons[0].weights[1] = Parameter(1.0)
        lin.neurons[0].bias = Parameter(0.0)
        x = [Value(3.0), Value(4.0)]
        y = lin(x)
        y[0].backward()
        # dy/dw0 = x0 = 3.0, dy/dw1 = x1 = 4.0
        assert lin.neurons[0].weights[0].grad == pytest.approx(3.0)
        assert lin.neurons[0].weights[1].grad == pytest.approx(4.0)

    def test_bias_gradient(self):
        """dy_j/db_j = 1."""
        lin = Linear(2, 1)
        lin.neurons[0].bias = Parameter(0.0)
        x = [Value(3.0), Value(4.0)]
        y = lin(x)
        y[0].backward()
        assert lin.neurons[0].bias.grad == pytest.approx(1.0)

    def test_input_gradient(self):
        """dy_j/dx_i = w_{j,i}."""
        lin = Linear(2, 1)
        lin.neurons[0].weights[0] = Parameter(2.0)
        lin.neurons[0].weights[1] = Parameter(3.0)
        lin.neurons[0].bias = Parameter(0.0)
        x = [Value(1.0), Value(1.0)]
        y = lin(x)
        y[0].backward()
        assert x[0].grad == pytest.approx(2.0)
        assert x[1].grad == pytest.approx(3.0)

    def test_multi_output_independent_backward(self):
        """Two outputs, backward on one should not affect the other's weights."""
        lin = Linear(2, 2)
        lin.neurons[0].weights[0] = Parameter(1.0)
        lin.neurons[0].bias = Parameter(0.0)
        lin.neurons[1].weights[0] = Parameter(1.0)
        lin.neurons[1].bias = Parameter(0.0)
        x = [Value(1.0), Value(1.0)]
        y = lin(x)
        y[0].backward()
        # Neuron 0 weights should have grads, neuron 1 should be zero
        assert lin.neurons[0].weights[0].grad == pytest.approx(1.0)
        assert lin.neurons[1].weights[0].grad == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. ReLU and Tanh layers
# ---------------------------------------------------------------------------

class TestReLULayer:
    def test_positive(self):
        act = ReLU()
        out = act(Value(5.0))
        assert out.data == pytest.approx(5.0)

    def test_negative(self):
        act = ReLU()
        out = act(Value(-3.0))
        assert out.data == pytest.approx(0.0)

    def test_zero(self):
        act = ReLU()
        out = act(Value(0.0))
        assert out.data == pytest.approx(0.0)

    def test_gradient(self):
        act = ReLU()
        x = Value(2.0)
        out = act(x)
        out.backward()
        assert x.grad == pytest.approx(1.0)

    def test_gradient_negative(self):
        act = ReLU()
        x = Value(-2.0)
        out = act(x)
        out.backward()
        assert x.grad == pytest.approx(0.0)


class TestTanhLayer:
    def test_zero(self):
        act = Tanh()
        out = act(Value(0.0))
        assert out.data == pytest.approx(0.0)

    def test_positive(self):
        act = Tanh()
        out = act(Value(1.0))
        assert out.data == pytest.approx(math.tanh(1.0))

    def test_gradient(self):
        act = Tanh()
        x = Value(0.5)
        out = act(x)
        out.backward()
        expected = 1.0 - math.tanh(0.5) ** 2
        assert x.grad == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5. Gradient checking for layers
# ---------------------------------------------------------------------------

class TestNeuronGradientCheck:
    def test_single_neuron(self):
        """Numerical gradient matches analytical for a single neuron."""
        n = Neuron(3)
        x = [Value(1.0), Value(2.0), Value(3.0)]
        out = n(x)
        out.backward()
        # Check weight gradients
        params = n.parameters()
        results = gradient_check(
            lambda: n([Value(1.0), Value(2.0), Value(3.0)]).data,
            params
        )
        assert all(r['passed'] for r in results)


class TestLinearGradientCheck:
    def test_single_output(self):
        """Numerical gradient matches analytical for Linear(3, 1)."""
        lin = Linear(3, 1)
        x = [Value(1.0), Value(2.0), Value(3.0)]
        y = lin(x)
        y[0].backward()
        params = lin.parameters()
        results = gradient_check(
            lambda: lin([Value(1.0), Value(2.0), Value(3.0)])[0].data,
            params
        )
        assert all(r['passed'] for r in results)

    def test_multi_output(self):
        """Numerical gradient matches analytical for Linear(2, 3)."""
        lin = Linear(2, 3)
        x = [Value(1.0), Value(2.0)]
        y = lin(x)
        # Backprop from all outputs
        for yi in y:
            yi.backward()
        params = lin.parameters()
        results = gradient_check(
            lambda: sum(lin([Value(1.0), Value(2.0)])[j].data for j in range(3)),
            params
        )
        assert all(r['passed'] for r in results)

    def test_relu_linear_composed(self):
        """Numerical gradient for ReLU(Linear(x))."""
        lin = Linear(2, 1)
        act = ReLU()
        x = [Value(1.0), Value(-1.0)]
        y = lin(x)
        out = act(y[0])
        out.backward()
        params = lin.parameters()
        results = gradient_check(
            lambda: act(lin([Value(1.0), Value(-1.0)])[0]).data,
            params
        )
        assert all(r['passed'] for r in results)


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_inputs(self):
        n = Neuron(2)
        x = [Value(0.0), Value(0.0)]
        out = n(x)
        # Should be just bias
        assert out.data == pytest.approx(n.bias.data)

    def test_repeated_forward_backward(self):
        lin = Linear(2, 1)
        x = [Value(1.0), Value(2.0)]
        y = lin(x)
        y[0].backward()
        g1 = [p.grad for p in lin.parameters()]
        # Second pass
        y = lin(x)
        y[0].backward()
        g2 = [p.grad for p in lin.parameters()]
        # D19: should accumulate
        for g1i, g2i in zip(g1, g2):
            assert g2i == pytest.approx(2 * g1i)

    def test_zero_grad_then_backward(self):
        lin = Linear(2, 1)
        x = [Value(1.0), Value(2.0)]
        y = lin(x)
        y[0].backward()
        lin.zero_grad()
        # All grads should be zero
        assert all(p.grad == 0.0 for p in lin.parameters())
        # Backward again should work
        y = lin(x)
        y[0].backward()
        assert all(p.grad != 0.0 for p in lin.parameters())

    def test_deterministic_with_seed(self):
        """Two Neurons with same construction should have same initial weights."""
        # Our deterministic initialization uses a formula based on index
        n1 = Neuron(5)
        n2 = Neuron(5)
        for w1, w2 in zip(n1.weights, n2.weights):
            assert w1.data == pytest.approx(w2.data)
        assert n1.bias.data == pytest.approx(n2.bias.data)
