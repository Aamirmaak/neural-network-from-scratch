"""API smoke tests — verify clean imports and full training workflow."""

import sys
import pytest


class TestCleanImport:
    """Verify all public symbols import cleanly."""

    def test_version(self):
        import neuralearn
        assert isinstance(neuralearn.__version__, str)
        parts = neuralearn.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_all_symbols_importable(self):
        import neuralearn
        for name in neuralearn.__all__:
            assert hasattr(neuralearn, name), f"{name} in __all__ but not importable"

    def test_all_matches_exports(self):
        import neuralearn
        expected = {
            "Value", "Parameter",
            "Module", "Neuron", "Linear", "ReLU", "Tanh", "Sigmoid",
            "mse_loss", "binary_cross_entropy",
            "SGD", "MomentumSGD", "Adam",
            "Trainer",
            "Dataset", "DataLoader",
            "numerical_grad", "gradient_check",
        }
        assert set(neuralearn.__all__) == expected

    def test_lazy_viz_import(self):
        from neuralearn import plot_loss
        assert callable(plot_loss)


class TestFullTrainingWorkflow:
    """End-to-end training with all components."""

    def test_binary_classification(self):
        from neuralearn import (
            Value, Linear, Tanh, Sigmoid,
            binary_cross_entropy, SGD, Trainer,
        )

        class TinyNet:
            def __init__(self):
                self.layer1 = Linear(2, 4)
                self.act = Tanh()
                self.layer2 = Linear(4, 1)
                self.sig = Sigmoid()

            def __call__(self, x):
                h = [self.act(v) for v in self.layer1(x)]
                out = self.layer2(h)
                return [self.sig(v) for v in out]

            def parameters(self):
                return self.layer1.parameters() + self.layer2.parameters()

            def zero_grad(self):
                self.layer1.zero_grad()
                self.layer2.zero_grad()

        net = TinyNet()
        opt = SGD(net.parameters(), lr=0.1)
        trainer = Trainer(net, binary_cross_entropy, opt)

        inputs = [[Value(0.0), Value(0.0)],
                   [Value(0.0), Value(1.0)],
                   [Value(1.0), Value(0.0)],
                   [Value(1.0), Value(1.0)]]
        targets = [[Value(0.0)], [Value(1.0)],
                    [Value(1.0)], [Value(0.0)]]

        history = trainer.fit(inputs, targets, epochs=500)
        assert history["loss"][-1] < history["loss"][0]

    def test_regression_with_dataloader(self):
        from neuralearn import (
            Value, Linear, Tanh,
            mse_loss, Adam, Trainer, Dataset, DataLoader,
        )

        class TinyNet:
            def __init__(self):
                self.layer = Linear(1, 1)
                self.act = Tanh()

            def __call__(self, x):
                return [self.act(v) for v in self.layer(x)]

            def parameters(self):
                return self.layer.parameters()

            def zero_grad(self):
                self.layer.zero_grad()

        ds = Dataset(
            [[float(i)] for i in range(10)],
            [[float(i * 2)] for i in range(10)],
        )
        loader = DataLoader(ds, batch_size=5, shuffle=False)

        net = TinyNet()
        opt = Adam(net.parameters(), lr=0.01)
        trainer = Trainer(net, mse_loss, opt)

        history = trainer.fit(loader, epochs=50)
        assert "loss" in history
        assert len(history["loss"]) == 50

    def test_gradient_check_passes(self):
        from neuralearn import Value, gradient_check

        a, b = Value(3.0), Value(4.0)
        out = a * b
        out.backward()
        results = gradient_check(lambda: a.data * b.data, [a, b])
        assert all(r["passed"] for r in results)
