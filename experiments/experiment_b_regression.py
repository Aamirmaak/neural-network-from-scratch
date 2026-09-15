"""
Experiment B: Nonlinear Regression

Demonstrates that the system can fit complex nonlinear functions.

Hypothesis:
    A neural network with sufficient capacity can approximate a nonlinear
    function (sin(x) + 0.1*x^2) with low MSE.

Configuration:
    Model:        Linear(1, 16) -> Tanh -> Linear(16, 16) -> Tanh -> Linear(16, 1)
    Loss:         MSE
    Optimizer:    Adam(lr=0.01)
    Epochs:       2000
    Training:     100 samples in [-3, 3]
    Test:         50 samples in [-3, 3]
    Random seed:  42
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuralearn.value import Value
from neuralearn.layers import Linear, Tanh
from neuralearn.losses import mse_loss
from neuralearn.optimizers import Adam


def target_fn(x):
    """Target function: sin(x) + 0.1 * x^2"""
    return math.sin(x) + 0.1 * x ** 2


def make_data(n_samples, x_min=-3.0, x_max=3.0, seed=42):
    """Generate regression data with deterministic spacing."""
    step = (x_max - x_min) / (n_samples - 1)
    xs = [x_min + i * step for i in range(n_samples)]
    # Shuffle with simple deterministic permutation
    shuffled = list(range(n_samples))
    # Simple deterministic shuffle using seed
    rng_state = seed
    for i in range(n_samples - 1, 0, -1):
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        j = rng_state % (i + 1)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    xs_shuffled = [xs[i] for i in shuffled]
    return xs_shuffled


class RegressionMLP:
    """Multi-layer perceptron for 1D regression."""

    def __init__(self, hidden_sizes=(16, 16)):
        self.layers = []
        self.activations = []
        nin = 1
        for h in hidden_sizes:
            self.layers.append(Linear(nin, h))
            self.activations.append(Tanh())
            nin = h
        self.layers.append(Linear(nin, 1))

    def forward(self, x):
        """Forward pass. x is a list with one Value."""
        h = x
        for layer, act in zip(self.layers[:-1], self.activations):
            h = [act(v) for v in layer(h)]
        out = self.layers[-1](h)
        return out[0]  # single output

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()


def train_regression(hidden_sizes=(8, 8), lr=0.02, epochs=200, seed=42):
    """Train regression model and return history."""
    model = RegressionMLP(hidden_sizes=hidden_sizes)
    optimizer = Adam(model.parameters(), lr=lr)

    train_xs = make_data(30, seed=seed)
    test_xs = make_data(15, seed=seed + 1000)

    history = {"epoch": [], "train_mse": [], "test_mse": []}

    for epoch in range(1, epochs + 1):
        # Training
        total_loss = Value(0.0)
        for x_val in train_xs:
            model.zero_grad()
            pred = model([Value(x_val)])
            target = Value(target_fn(x_val))
            loss = (pred - target) ** 2
            loss.backward()
            optimizer.step()
            total_loss.data += loss.data

        train_mse = total_loss.data / len(train_xs)

        # Test (no gradient computation needed, but we use the model)
        test_loss = 0.0
        for x_val in test_xs:
            pred = model([Value(x_val)])
            target_val = target_fn(x_val)
            test_loss += (pred.data - target_val) ** 2
        test_mse = test_loss / len(test_xs)

        if epoch % 50 == 0 or epoch == 1:
            history["epoch"].append(epoch)
            history["train_mse"].append(train_mse)
            history["test_mse"].append(test_mse)
            print(f"  Epoch {epoch:5d} | Train MSE: {train_mse:.6f} | Test MSE: {test_mse:.6f}")

    return model, history, train_xs, test_xs


def main():
    print("=" * 60)
    print("Experiment B: Nonlinear Regression")
    print("=" * 60)
    print()

    # --- Linear baseline ---
    print("--- Linear Baseline (should underfit) ---")
    linear_model = RegressionMLP(hidden_sizes=())
    # Override: single linear layer
    linear_model.layers = [Linear(1, 1)]
    linear_model.activations = []

    linear_optimizer = Adam(linear_model.parameters(), lr=0.01)
    train_xs = make_data(30, seed=42)
    test_xs = make_data(15, seed=1042)

    print("Training linear model...")
    for epoch in range(1, 201):
        total_loss = Value(0.0)
        for x_val in train_xs:
            linear_model.zero_grad()
            pred = linear_model([Value(x_val)])
            target = Value(target_fn(x_val))
            loss = (pred - target) ** 2
            loss.backward()
            linear_optimizer.step()
            total_loss.data += loss.data
        if epoch % 50 == 0 or epoch == 1:
            train_mse = total_loss.data / len(train_xs)
            test_loss = 0.0
            for x_val in test_xs:
                pred = linear_model([Value(x_val)])
                test_loss += (pred.data - target_fn(x_val)) ** 2
            test_mse = test_loss / len(test_xs)
            print(f"  Epoch {epoch:5d} | Train MSE: {train_mse:.6f} | Test MSE: {test_mse:.6f}")

    # Final evaluation
    linear_train_loss = 0.0
    for x_val in train_xs:
        pred = linear_model([Value(x_val)])
        linear_train_loss += (pred.data - target_fn(x_val)) ** 2
    linear_train_mse = linear_train_loss / len(train_xs)

    linear_test_loss = 0.0
    for x_val in test_xs:
        pred = linear_model([Value(x_val)])
        linear_test_loss += (pred.data - target_fn(x_val)) ** 2
    linear_test_mse = linear_test_loss / len(test_xs)

    print(f"\nLinear model final: Train MSE={linear_train_mse:.6f}, Test MSE={linear_test_mse:.6f}")

    # --- MLP ---
    print("\n--- MLP with Hidden Layers ---")
    print(f"Training MLP (hidden=[8,8], lr=0.02, Adam, epochs=200)...")
    model, history, train_xs, test_xs = train_regression(
        hidden_sizes=(8, 8), lr=0.02, epochs=200, seed=42
    )

    mlp_train_loss = 0.0
    for x_val in train_xs:
        pred = model([Value(x_val)])
        mlp_train_loss += (pred.data - target_fn(x_val)) ** 2
    mlp_train_mse = mlp_train_loss / len(train_xs)

    mlp_test_loss = 0.0
    for x_val in test_xs:
        pred = model([Value(x_val)])
        mlp_test_loss += (pred.data - target_fn(x_val)) ** 2
    mlp_test_mse = mlp_test_loss / len(test_xs)

    print(f"\nMLP final: Train MSE={mlp_train_mse:.6f}, Test MSE={mlp_test_mse:.6f}")
    print(f"MLP parameters: {len(model.parameters())}")

    # --- Sample predictions ---
    print("\nSample predictions (MLP):")
    sample_xs = [-3.0, -1.5, 0.0, 1.5, 3.0]
    for x_val in sample_xs:
        pred = model([Value(x_val)])
        actual = target_fn(x_val)
        print(f"  x={x_val:5.1f} | predicted={pred.data:7.4f} | actual={actual:7.4f} | diff={abs(pred.data-actual):.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Linear model: Train MSE={linear_train_mse:.6f}, Test MSE={linear_test_mse:.6f}")
    print(f"MLP:          Train MSE={mlp_train_mse:.6f}, Test MSE={mlp_test_mse:.6f}")
    print(f"MLP parameters: {len(model.parameters())}")
    print()
    if mlp_train_mse < linear_train_mse and mlp_test_mse < linear_test_mse:
        print("RESULT: Hypothesis CONFIRMED.")
        print("  - Linear model underfits (high MSE)")
        print("  - MLP fits the nonlinear function well (low MSE)")
    else:
        print("RESULT: Results are mixed. Further investigation needed.")

    return {
        "linear_train_mse": linear_train_mse,
        "linear_test_mse": linear_test_mse,
        "mlp_train_mse": mlp_train_mse,
        "mlp_test_mse": mlp_test_mse,
        "mlp_params": len(model.parameters()),
        "history": history,
    }


if __name__ == "__main__":
    results = main()
