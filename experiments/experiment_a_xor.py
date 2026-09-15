"""
Experiment A: XOR Classification

Demonstrates that a neural network with a hidden layer can learn
the XOR function, while a linear model cannot.

Hypothesis:
    A neural network with at least one hidden layer can learn XOR
    with near-perfect accuracy.

Configuration:
    Model:        Linear(2, hidden) -> Tanh -> Linear(hidden, 1) -> Sigmoid
    Loss:         Binary Cross-Entropy
    Optimizer:    Adam(lr=0.01)
    Epochs:       1000
    Random seed:  42
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuralearn.value import Value
from neuralearn.layers import Linear, Tanh, Sigmoid
from neuralearn.losses import binary_cross_entropy
from neuralearn.optimizers import Adam


def make_xor_data():
    """Return XOR dataset as list of (input, target) pairs."""
    return [
        ([Value(0.0), Value(0.0)], [Value(0.0)]),
        ([Value(0.0), Value(1.0)], [Value(1.0)]),
        ([Value(1.0), Value(0.0)], [Value(1.0)]),
        ([Value(1.0), Value(1.0)], [Value(0.0)]),
    ]


class MLP:
    """Simple multi-layer perceptron for XOR."""

    def __init__(self, hidden_size=8, seed=42):
        self.layer1 = Linear(2, hidden_size)
        self.act1 = Tanh()
        self.layer2 = Linear(hidden_size, 1)
        self.act2 = Sigmoid()

    def forward(self, x):
        h = [self.act1(v) for v in self.layer1(x)]
        out = self.layer2(h)
        return [self.act2(v) for v in out]

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters()

    def zero_grad(self):
        self.layer1.zero_grad()
        self.layer2.zero_grad()


def train_xor(hidden_size=8, lr=0.01, epochs=1000):
    """Train XOR classifier and return training history."""
    model = MLP(hidden_size=hidden_size)
    optimizer = Adam(model.parameters(), lr=lr)
    data = make_xor_data()

    history = {"epoch": [], "loss": [], "accuracy": []}

    for epoch in range(1, epochs + 1):
        total_loss = Value(0.0)
        correct = 0

        for inputs, targets in data:
            model.zero_grad()
            preds = model(inputs)
            loss = binary_cross_entropy(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss.data += loss.data
            predicted = 1.0 if preds[0].data >= 0.5 else 0.0
            if predicted == targets[0].data:
                correct += 1

        avg_loss = total_loss.data / len(data)
        acc = correct / len(data) * 100

        if epoch % 100 == 0 or epoch == 1:
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)
            history["accuracy"].append(acc)
            print(f"  Epoch {epoch:4d} | Loss: {avg_loss:.6f} | Accuracy: {acc:.1f}%")

    return model, history


def evaluate_xor(model, data):
    """Evaluate model on XOR data."""
    correct = 0
    results = []
    for inputs, targets in data:
        preds = model(inputs)
        predicted = 1.0 if preds[0].data >= 0.5 else 0.0
        match = predicted == targets[0].data
        if match:
            correct += 1
        results.append({
            "input": [v.data for v in inputs],
            "target": targets[0].data,
            "prediction": preds[0].data,
            "predicted_class": predicted,
            "correct": match,
        })
    return correct / len(data), results


def main():
    print("=" * 60)
    print("Experiment A: XOR Classification")
    print("=" * 60)
    print()

    # --- Linear baseline ---
    print("--- Linear Baseline (should fail) ---")

    class LinearModel:
        def __init__(self):
            self.linear = Linear(2, 1)
            self.sigmoid = Sigmoid()

        def forward(self, x):
            out = self.linear(x)
            return [self.sigmoid(v) for v in out]

        def __call__(self, x):
            return self.forward(x)

        def parameters(self):
            return self.linear.parameters()

        def zero_grad(self):
            self.linear.zero_grad()

    linear = LinearModel()
    linear_optimizer = Adam(linear.parameters(), lr=0.01)
    data = make_xor_data()

    print("Training linear model...")
    for epoch in range(1, 1001):
        total_loss = Value(0.0)
        correct = 0
        for inputs, targets in data:
            linear.zero_grad()
            preds = linear(inputs)
            loss = binary_cross_entropy(preds, targets)
            loss.backward()
            linear_optimizer.step()
            total_loss.data += loss.data
            predicted = 1.0 if preds[0].data >= 0.5 else 0.0
            if predicted == targets[0].data:
                correct += 1
        if epoch % 200 == 0 or epoch == 1:
            avg_loss = total_loss.data / len(data)
            acc = correct / len(data) * 100
            print(f"  Epoch {epoch:4d} | Loss: {avg_loss:.6f} | Accuracy: {acc:.1f}%")

    linear_acc, linear_results = evaluate_xor(linear, data)
    print(f"\nLinear model final accuracy: {linear_acc * 100:.1f}%")
    print("Linear predictions:")
    for r in linear_results:
        print(f"  {r['input']} -> {r['prediction']:.4f} (target: {r['target']:.0f})")

    # --- MLP ---
    print("\n--- MLP with Hidden Layer ---")
    print(f"Training MLP (hidden=8, lr=0.01, Adam, epochs=1000)...")
    model, history = train_xor(hidden_size=8, lr=0.01, epochs=1000)

    mlp_acc, mlp_results = evaluate_xor(model, data)
    print(f"\nMLP final accuracy: {mlp_acc * 100:.1f}%")
    print("MLP predictions:")
    for r in mlp_results:
        print(f"  {r['input']} -> {r['prediction']:.4f} (target: {r['target']:.0f})")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Linear model accuracy: {linear_acc * 100:.1f}%")
    print(f"MLP accuracy:          {mlp_acc * 100:.1f}%")
    print(f"MLP parameters:        {len(model.parameters())}")
    print()
    if mlp_acc == 1.0 and linear_acc < 1.0:
        print("RESULT: Hypothesis CONFIRMED.")
        print("  - Linear model cannot learn XOR (as expected)")
        print("  - MLP with hidden layer learns XOR perfectly")
    elif mlp_acc == 1.0:
        print("RESULT: MLP learns XOR, but linear model also succeeded (unexpected).")
    else:
        print("RESULT: Hypothesis NOT confirmed. MLP failed to learn XOR.")

    return {
        "linear_accuracy": linear_acc,
        "mlp_accuracy": mlp_acc,
        "mlp_params": len(model.parameters()),
        "history": history,
        "mlp_results": mlp_results,
        "linear_results": linear_results,
    }


if __name__ == "__main__":
    results = main()
