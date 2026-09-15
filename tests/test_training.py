"""
Tests for the training engine (Stage 7).

Tests cover:
- Trainer construction and validation
- Training execution and epoch behavior
- Loss history structure and values
- Parameter updates during training
- Gradient reset behavior (D19 verification)
- Evaluation without parameter modification
- Input validation and error handling
- Simple learning experiments (linear, multi-layer, BCE)
"""

import pytest

from neuralearn.value import Value
from neuralearn.parameter import Parameter
from neuralearn.layers import Linear, Module, ReLU, Tanh
from neuralearn.losses import mse_loss, binary_cross_entropy
from neuralearn.optimizers import SGD, MomentumSGD, Adam
from neuralearn.training import Trainer


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_construction(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)
        assert trainer.model is layer
        assert trainer.loss_fn is mse_loss
        assert trainer.optimizer is opt

    def test_none_model_raises(self):
        with pytest.raises(TypeError):
            Trainer(None, mse_loss, SGD([], lr=0.01))

    def test_none_loss_fn_raises(self):
        with pytest.raises(TypeError):
            Trainer(Linear(1, 1), None, SGD([], lr=0.01))

    def test_none_optimizer_raises(self):
        with pytest.raises(TypeError):
            Trainer(Linear(1, 1), mse_loss, None)

    def test_repr(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)
        r = repr(trainer)
        assert "Linear" in r
        assert "mse_loss" in r
        assert "SGD" in r


# ---------------------------------------------------------------------------
# 2. Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def _make_trainer(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        return Trainer(layer, mse_loss, opt)

    def test_zero_epochs_raises(self):
        trainer = self._make_trainer()
        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]
        with pytest.raises(ValueError, match="epochs"):
            trainer.fit(inputs, targets, epochs=0)

    def test_negative_epochs_raises(self):
        trainer = self._make_trainer()
        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]
        with pytest.raises(ValueError, match="epochs"):
            trainer.fit(inputs, targets, epochs=-1)

    def test_empty_inputs_raises_fit(self):
        trainer = self._make_trainer()
        with pytest.raises(ValueError, match="empty"):
            trainer.fit([], [], epochs=1)

    def test_empty_inputs_raises_evaluate(self):
        trainer = self._make_trainer()
        with pytest.raises(ValueError, match="empty"):
            trainer.evaluate([], [])

    def test_mismatched_lengths_raises_fit(self):
        trainer = self._make_trainer()
        inputs = [[Value(1.0)], [Value(2.0)]]
        targets = [[Value(3.0)]]
        with pytest.raises(ValueError, match="same length"):
            trainer.fit(inputs, targets, epochs=1)

    def test_mismatched_lengths_raises_evaluate(self):
        trainer = self._make_trainer()
        inputs = [[Value(1.0)]]
        targets = [[Value(1.0)], [Value(2.0)]]
        with pytest.raises(ValueError, match="same length"):
            trainer.evaluate(inputs, targets)


# ---------------------------------------------------------------------------
# 3. Training Execution
# ---------------------------------------------------------------------------

class TestTrainingExecution:
    def test_basic_training_runs(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]
        history = trainer.fit(inputs, targets, epochs=5)
        assert history is not None

    def test_correct_epoch_count(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]
        history = trainer.fit(inputs, targets, epochs=10)
        assert len(history["loss"]) == 10

    def test_single_epoch(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]
        history = trainer.fit(inputs, targets, epochs=1)
        assert len(history["loss"]) == 1

    def test_history_keys(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]
        history = trainer.fit(inputs, targets, epochs=5)
        assert "loss" in history


# ---------------------------------------------------------------------------
# 4. Loss History
# ---------------------------------------------------------------------------

class TestLossHistory:
    def test_loss_values_are_floats(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]
        history = trainer.fit(inputs, targets, epochs=5)
        for loss_val in history["loss"]:
            assert isinstance(loss_val, float)

    def test_loss_is_positive(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]
        history = trainer.fit(inputs, targets, epochs=5)
        for loss_val in history["loss"]:
            assert loss_val >= 0.0

    def test_loss_values_differ_across_epochs(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]
        history = trainer.fit(inputs, targets, epochs=10)
        # With training, at least some loss values should differ
        assert len(set(history["loss"])) > 1


# ---------------------------------------------------------------------------
# 5. Parameter Updates
# ---------------------------------------------------------------------------

class TestParameterUpdates:
    def test_parameters_change_during_training(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]

        # Record initial parameters
        initial_params = [p.data for p in layer.parameters()]

        trainer.fit(inputs, targets, epochs=5)

        # At least some parameters should have changed
        final_params = [p.data for p in layer.parameters()]
        assert any(
            abs(i - f) > 1e-10
            for i, f in zip(initial_params, final_params)
        )

    def test_multiple_samples_all_parameters_update(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [
            [Value(1.0), Value(2.0)],
            [Value(3.0), Value(4.0)],
            [Value(5.0), Value(6.0)],
        ]
        targets = [[Value(5.0)], [Value(11.0)], [Value(17.0)]]

        initial_params = [p.data for p in layer.parameters()]
        trainer.fit(inputs, targets, epochs=10)
        final_params = [p.data for p in layer.parameters()]

        assert any(
            abs(i - f) > 1e-10
            for i, f in zip(initial_params, final_params)
        )


# ---------------------------------------------------------------------------
# 6. Gradient Reset Behavior (D19 Verification)
# ---------------------------------------------------------------------------

class TestGradientReset:
    def test_gradients_are_zero_after_training(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]
        trainer.fit(inputs, targets, epochs=5)

        # After zero_grad in the last iteration, all parameter grads should be 0
        for p in layer.parameters():
            assert p.grad == 0.0

    def test_no_stale_gradient_accumulation(self):
        """Verify D19: repeated training iterations don't accumulate stale gradients."""
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]

        # Train one epoch, record final grads (should be 0 after zero_grad)
        trainer.fit(inputs, targets, epochs=1)
        grads_after_first = [p.grad for p in layer.parameters()]
        assert all(g == 0.0 for g in grads_after_first)

        # Train another epoch — grads should still be 0 after
        trainer.fit(inputs, targets, epochs=1)
        grads_after_second = [p.grad for p in layer.parameters()]
        assert all(g == 0.0 for g in grads_after_second)


# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_evaluate_returns_history(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]
        history = trainer.evaluate(inputs, targets)
        assert "loss" in history
        assert len(history["loss"]) == 1

    def test_evaluate_loss_is_float(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]
        history = trainer.evaluate(inputs, targets)
        assert isinstance(history["loss"][0], float)

    def test_evaluate_does_not_update_parameters(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]

        # Record parameters before evaluation
        pre_params = [p.data for p in layer.parameters()]

        trainer.evaluate(inputs, targets)

        # Parameters must be unchanged
        post_params = [p.data for p in layer.parameters()]
        for pre, post in zip(pre_params, post_params):
            assert pre == post

    def test_evaluate_does_not_update_gradients(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]

        # Ensure grads are zero before
        for p in layer.parameters():
            p.zero_grad()

        trainer.evaluate(inputs, targets)

        # Gradients should still be zero (no backward called)
        for p in layer.parameters():
            assert p.grad == 0.0

    def test_evaluate_consistent_results(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]

        h1 = trainer.evaluate(inputs, targets)
        h2 = trainer.evaluate(inputs, targets)
        # Same model state → same loss
        assert h1["loss"][0] == h2["loss"][0]

    def test_evaluate_before_and_after_training(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0), Value(2.0)]]
        targets = [[Value(3.0)]]

        loss_before = trainer.evaluate(inputs, targets)["loss"][0]
        trainer.fit(inputs, targets, epochs=50)
        loss_after = trainer.evaluate(inputs, targets)["loss"][0]

        # Loss should decrease after training on a learnable problem
        assert loss_after < loss_before


# ---------------------------------------------------------------------------
# 8. Simple Learning Experiment — Linear y = 2x + 1
# ---------------------------------------------------------------------------

class TestLinearLearning:
    def test_loss_decreases(self):
        """Train Linear(1,1) on y = 2x + 1 with MSE + SGD."""
        layer = Linear(1, 1)
        # Set known weights for reproducibility
        layer.neurons[0].weights[0] = Parameter(0.0)
        layer.neurons[0].bias = Parameter(0.0)

        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        # y = 2x + 1
        inputs = [[Value(x)] for x in [0.0, 1.0, 2.0, 3.0, 4.0]]
        targets = [[Value(2 * x + 1)] for x in [0.0, 1.0, 2.0, 3.0, 4.0]]

        history = trainer.fit(inputs, targets, epochs=200)

        # Loss should decrease
        assert history["loss"][-1] < history["loss"][0]

    def test_parameters_move_toward_solution(self):
        """Parameters should approach w=2, b=1."""
        layer = Linear(1, 1)
        layer.neurons[0].weights[0] = Parameter(0.0)
        layer.neurons[0].bias = Parameter(0.0)

        opt = SGD(layer.parameters(), lr=0.05)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(x)] for x in [0.0, 1.0, 2.0, 3.0]]
        targets = [[Value(2 * x + 1)] for x in [0.0, 1.0, 2.0, 3.0]]

        trainer.fit(inputs, targets, epochs=500)

        w = layer.neurons[0].weights[0].data
        b = layer.neurons[0].bias.data
        # Should be close to w=2, b=1 (not exact, but in the right direction)
        assert abs(w - 2.0) < 0.5
        assert abs(b - 1.0) < 0.5

    def test_final_loss_is_small(self):
        """After sufficient training, MSE should be small."""
        layer = Linear(1, 1)
        layer.neurons[0].weights[0] = Parameter(0.0)
        layer.neurons[0].bias = Parameter(0.0)

        opt = SGD(layer.parameters(), lr=0.05)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(x)] for x in [0.0, 1.0, 2.0, 3.0]]
        targets = [[Value(2 * x + 1)] for x in [0.0, 1.0, 2.0, 3.0]]

        history = trainer.fit(inputs, targets, epochs=500)
        # Final loss should be less than initial
        assert history["loss"][-1] < history["loss"][0] * 0.1


# ---------------------------------------------------------------------------
# 9. Multi-Layer Learning Experiment
# ---------------------------------------------------------------------------

class TestMultiLayerLearning:
    def test_two_layer_loss_decreases(self):
        """Linear(1, 4) → ReLU → Linear(4, 1) with MSE + Adam."""

        class TinyNet(Module):
            def __init__(self):
                super().__init__()
                self.layer1 = Linear(1, 4)
                self.act = ReLU()
                self.layer2 = Linear(4, 1)

            def forward(self, x):
                h = self.layer1(x)
                h = [self.act(v) for v in h]
                return self.layer2(h)

            def parameters(self):
                return self.layer1.parameters() + self.layer2.parameters()

        net = TinyNet()
        opt = Adam(net.parameters(), lr=0.01)
        trainer = Trainer(net, mse_loss, opt)

        # Simple function: y = x^2 (need ReLU for nonlinearity)
        inputs = [[Value(x)] for x in [0.0, 0.5, 1.0, 1.5, 2.0]]
        targets = [[Value(x ** 2)] for x in [0.0, 0.5, 1.0, 1.5, 2.0]]

        history = trainer.fit(inputs, targets, epochs=200)
        assert history["loss"][-1] < history["loss"][0]

    def test_multi_layer_with_tanh(self):
        """Linear(1, 4) → Tanh → Linear(4, 1) with MSE + MomentumSGD."""

        class TinyNet(Module):
            def __init__(self):
                super().__init__()
                self.layer1 = Linear(1, 4)
                self.act = Tanh()
                self.layer2 = Linear(4, 1)

            def forward(self, x):
                h = self.layer1(x)
                h = [self.act(v) for v in h]
                return self.layer2(h)

            def parameters(self):
                return self.layer1.parameters() + self.layer2.parameters()

        net = TinyNet()
        opt = MomentumSGD(net.parameters(), lr=0.01, momentum=0.9)
        trainer = Trainer(net, mse_loss, opt)

        inputs = [[Value(x)] for x in [0.0, 0.5, 1.0, 1.5, 2.0]]
        targets = [[Value(x ** 2)] for x in [0.0, 0.5, 1.0, 1.5, 2.0]]

        history = trainer.fit(inputs, targets, epochs=200)
        assert history["loss"][-1] < history["loss"][0]


# ---------------------------------------------------------------------------
# 10. BCE Training Experiment
# ---------------------------------------------------------------------------

class TestBCETraining:
    def test_bce_loss_decreases(self):
        """Train with BCE on a tiny binary classification problem.

        BCE expects probabilities in (0, 1). We apply sigmoid to raw
        Linear output before passing to BCE, which is the standard pattern.
        """
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.1)
        trainer = Trainer(layer, binary_cross_entropy, opt)

        def sigmoid(x):
            return 1.0 / (1.0 + (-x).exp())

        def model_with_sigmoid(x):
            raw = layer(x)
            return [sigmoid(v) for v in raw]

        # Temporarily replace model call for this test
        original_model = trainer.model
        trainer.model = model_with_sigmoid

        inputs = [
            [Value(1.0), Value(0.0)],
            [Value(0.0), Value(1.0)],
        ]
        targets = [
            [Value(1.0)],
            [Value(0.0)],
        ]

        history = trainer.fit(inputs, targets, epochs=200)
        assert history["loss"][-1] < history["loss"][0]

    def test_bce_loss_is_positive(self):
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, binary_cross_entropy, opt)

        inputs = [[Value(1.0), Value(0.0)], [Value(0.0), Value(1.0)]]
        targets = [[Value(1.0)], [Value(0.0)]]

        history = trainer.fit(inputs, targets, epochs=10)
        for loss_val in history["loss"]:
            assert loss_val >= 0.0


# ---------------------------------------------------------------------------
# 11. Multiple Optimizer Integration
# ---------------------------------------------------------------------------

class TestOptimizerIntegration:
    def test_momentum_sgd_trains(self):
        layer = Linear(1, 1)
        layer.neurons[0].weights[0] = Parameter(0.0)
        layer.neurons[0].bias = Parameter(0.0)
        opt = MomentumSGD(layer.parameters(), lr=0.05, momentum=0.9)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(x)] for x in [0.0, 1.0, 2.0]]
        targets = [[Value(3 * x + 1)] for x in [0.0, 1.0, 2.0]]

        history = trainer.fit(inputs, targets, epochs=200)
        assert history["loss"][-1] < history["loss"][0]

    def test_adam_trains(self):
        layer = Linear(1, 1)
        layer.neurons[0].weights[0] = Parameter(0.0)
        layer.neurons[0].bias = Parameter(0.0)
        opt = Adam(layer.parameters(), lr=0.05)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(x)] for x in [0.0, 1.0, 2.0]]
        targets = [[Value(3 * x + 1)] for x in [0.0, 1.0, 2.0]]

        history = trainer.fit(inputs, targets, epochs=200)
        assert history["loss"][-1] < history["loss"][0]


# ---------------------------------------------------------------------------
# 12. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_sample_single_epoch(self):
        layer = Linear(1, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(1.0)]]
        targets = [[Value(2.0)]]
        history = trainer.fit(inputs, targets, epochs=1)
        assert len(history["loss"]) == 1

    def test_many_samples(self):
        layer = Linear(1, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        trainer = Trainer(layer, mse_loss, opt)

        inputs = [[Value(float(i))] for i in range(20)]
        targets = [[Value(float(i) * 2)] for i in range(20)]
        history = trainer.fit(inputs, targets, epochs=5)
        assert len(history["loss"]) == 5

    def test_reproducible_with_same_init(self):
        """Two trainers with same init produce same training history."""
        def make_trainer():
            layer = Linear(1, 1)
            layer.neurons[0].weights[0] = Parameter(0.5)
            layer.neurons[0].bias = Parameter(0.3)
            opt = SGD(layer.parameters(), lr=0.01)
            return Trainer(layer, mse_loss, opt)

        t1 = make_trainer()
        t2 = make_trainer()

        inputs = [[Value(1.0)], [Value(2.0)]]
        targets = [[Value(3.0)], [Value(5.0)]]

        h1 = t1.fit(inputs, targets, epochs=10)
        h2 = t2.fit(inputs, targets, epochs=10)

        assert h1["loss"] == h2["loss"]
