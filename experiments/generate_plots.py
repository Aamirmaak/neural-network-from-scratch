"""
Generate visualizations for Stage 9 experiments.

Runs XOR classification and nonlinear regression experiments,
then produces plots saved to artifacts/plots/.

Usage:
    python experiments/generate_plots.py

Output:
    artifacts/plots/xor_loss_curve.png
    artifacts/plots/xor_accuracy_curve.png
    artifacts/plots/xor_predictions.png
    artifacts/plots/regression_train_test_loss.png
    artifacts/plots/regression_fit.png
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuralearn.value import Value
from neuralearn.layers import Linear, Tanh, Sigmoid
from neuralearn.losses import binary_cross_entropy
from neuralearn.optimizers import Adam
from neuralearn.visualization import (
    plot_loss,
    plot_accuracy,
    plot_xor_predictions,
    plot_train_test_curve,
    plot_regression_fit,
)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "plots")


# ======================================================================
# XOR helpers (from experiment_a_xor.py)
# ======================================================================

def make_xor_data():
    return [
        ([Value(0.0), Value(0.0)], [Value(0.0)]),
        ([Value(0.0), Value(1.0)], [Value(1.0)]),
        ([Value(1.0), Value(0.0)], [Value(1.0)]),
        ([Value(1.0), Value(1.0)], [Value(0.0)]),
    ]


class MLP:
    def __init__(self, hidden_size=8):
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


def evaluate_xor(model, data):
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


# ======================================================================
# Regression helpers (from experiment_b_regression.py)
# ======================================================================

def target_fn(x):
    return math.sin(x) + 0.1 * x ** 2


def make_reg_data(n_samples, x_min=-3.0, x_max=3.0, seed=42):
    step = (x_max - x_min) / (n_samples - 1)
    xs = [x_min + i * step for i in range(n_samples)]
    shuffled = list(range(n_samples))
    rng_state = seed
    for i in range(n_samples - 1, 0, -1):
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        j = rng_state % (i + 1)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return [xs[i] for i in shuffled]


class RegressionMLP:
    def __init__(self, hidden_sizes=(8, 8)):
        self.layers = []
        self.activations = []
        nin = 1
        for h in hidden_sizes:
            self.layers.append(Linear(nin, h))
            self.activations.append(Tanh())
            nin = h
        self.layers.append(Linear(nin, 1))

    def forward(self, x):
        h = x
        for layer, act in zip(self.layers[:-1], self.activations):
            h = [act(v) for v in layer(h)]
        out = self.layers[-1](h)
        return out[0]

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


# ======================================================================
# Main: run experiments and generate plots
# ======================================================================

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    data = make_xor_data()

    # ---- XOR: Linear baseline ----
    print("Training XOR linear baseline...")
    linear = LinearModel()
    linear_opt = Adam(linear.parameters(), lr=0.01)
    for epoch in range(1, 1001):
        for inputs, targets in data:
            linear.zero_grad()
            preds = linear(inputs)
            loss = binary_cross_entropy(preds, targets)
            loss.backward()
            linear_opt.step()

    linear_acc, linear_results = evaluate_xor(linear, data)
    print(f"  Linear accuracy: {linear_acc*100:.1f}%")

    # ---- XOR: MLP ----
    print("Training XOR MLP...")
    model = MLP(hidden_size=8)
    optimizer = Adam(model.parameters(), lr=0.01)
    history = {"epoch": [], "loss": [], "accuracy": []}

    for epoch in range(1, 1001):
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

        if epoch % 50 == 0 or epoch == 1:
            avg_loss = total_loss.data / len(data)
            acc = correct / len(data) * 100
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)
            history["accuracy"].append(acc)

    mlp_acc, mlp_results = evaluate_xor(model, data)
    print(f"  MLP accuracy: {mlp_acc*100:.1f}%")

    # ---- XOR plots ----
    print("Generating XOR plots...")
    p1 = plot_loss(history, title="XOR Classification — Training Loss",
                   save_path=os.path.join(PLOT_DIR, "xor_loss_curve.png"))
    print(f"  Saved: {p1}")

    p2 = plot_accuracy(history, title="XOR Classification — Training Accuracy",
                       save_path=os.path.join(PLOT_DIR, "xor_accuracy_curve.png"))
    print(f"  Saved: {p2}")

    p3 = plot_xor_predictions(mlp_results, title="XOR — MLP Predictions",
                              save_path=os.path.join(PLOT_DIR, "xor_predictions.png"))
    print(f"  Saved: {p3}")

    # ---- Regression: Linear baseline ----
    print("\nTraining regression linear baseline...")
    lin_reg = RegressionMLP(hidden_sizes=())
    lin_reg.layers = [Linear(1, 1)]
    lin_reg.activations = []
    lin_reg_opt = Adam(lin_reg.parameters(), lr=0.01)
    train_xs = make_reg_data(30, seed=42)
    test_xs = make_reg_data(15, seed=1042)

    for epoch in range(1, 201):
        for x_val in train_xs:
            lin_reg.zero_grad()
            pred = lin_reg([Value(x_val)])
            target = Value(target_fn(x_val))
            loss = (pred - target) ** 2
            loss.backward()
            lin_reg_opt.step()

    # ---- Regression: MLP ----
    print("Training regression MLP...")
    reg_model = RegressionMLP(hidden_sizes=(8, 8))
    reg_opt = Adam(reg_model.parameters(), lr=0.02)
    reg_history = {"epoch": [], "train_mse": [], "test_mse": []}

    for epoch in range(1, 201):
        total_loss = Value(0.0)
        for x_val in train_xs:
            reg_model.zero_grad()
            pred = reg_model([Value(x_val)])
            target = Value(target_fn(x_val))
            loss = (pred - target) ** 2
            loss.backward()
            reg_opt.step()
            total_loss.data += loss.data

        train_mse = total_loss.data / len(train_xs)
        test_loss = 0.0
        for x_val in test_xs:
            pred = reg_model([Value(x_val)])
            test_loss += (pred.data - target_fn(x_val)) ** 2
        test_mse = test_loss / len(test_xs)

        if epoch % 25 == 0 or epoch == 1:
            reg_history["epoch"].append(epoch)
            reg_history["train_mse"].append(train_mse)
            reg_history["test_mse"].append(test_mse)

    print(f"  Final train MSE: {reg_history['train_mse'][-1]:.6f}")
    print(f"  Final test MSE:  {reg_history['test_mse'][-1]:.6f}")

    # ---- Regression plots ----
    print("Generating regression plots...")
    p4 = plot_train_test_curve(
        reg_history,
        title="Nonlinear Regression — Train vs Test MSE",
        save_path=os.path.join(PLOT_DIR, "regression_train_test_loss.png"),
    )
    print(f"  Saved: {p4}")

    # Collect predictions for the fit plot
    test_preds = []
    for x_val in test_xs:
        pred = reg_model([Value(x_val)])
        test_preds.append(pred.data)

    # Sort test data by x for clean plotting
    test_sorted = sorted(zip(test_xs, [target_fn(x) for x in test_xs], test_preds),
                          key=lambda t: t[0])
    x_test_sorted = [t[0] for t in test_sorted]
    y_test_sorted = [t[1] for t in test_sorted]
    pred_sorted = [t[2] for t in test_sorted]

    # Sort train data by x
    train_sorted = sorted(zip(train_xs, [target_fn(x) for x in train_xs]),
                           key=lambda t: t[0])
    x_train_sorted = [t[0] for t in train_sorted]
    y_train_sorted = [t[1] for t in train_sorted]

    p5 = plot_regression_fit(
        x_train_sorted, y_train_sorted,
        x_test_sorted, y_test_sorted, pred_sorted,
        target_fn=target_fn,
        title="Nonlinear Regression — MLP Fit",
        save_path=os.path.join(PLOT_DIR, "regression_fit.png"),
    )
    print(f"  Saved: {p5}")

    print(f"\nAll plots saved to: {PLOT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
