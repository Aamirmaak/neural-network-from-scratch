"""
Lightweight command-line interface for neuralearn.

Usage:
    neuralearn --version
    neuralearn --help
    neuralearn info
    neuralearn demo
    neuralearn demo --plot [--output DIR]
    neuralearn example
"""

from __future__ import annotations

import argparse
import sys


def _get_version() -> str:
    """Return the installed package version."""
    from neuralearn import __version__
    return __version__


def cmd_version(args: argparse.Namespace) -> int:
    """Print the package version."""
    print(_get_version())
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Print project information."""
    import neuralearn

    lines = [
        f"neuralearn {neuralearn.__version__}",
        "A neural network framework built from first principles",
        "",
        "Components:",
        "  Value            — scalar with gradient tracking",
        "  Parameter        — trainable value",
        "  Linear, ReLU, Tanh, Sigmoid — layers",
        "  mse_loss, binary_cross_entropy — losses",
        "  SGD, MomentumSGD, Adam — optimizers",
        "  Trainer          — training loop",
        "  Dataset, DataLoader — data utilities",
        "  numerical_grad, gradient_check — validation",
        f"  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    ]
    print("\n".join(lines))
    return 0


def _build_xor_demo_model(hidden_size: int = 8):
    """Build a tiny MLP for XOR classification."""
    from neuralearn.layers import Linear, Tanh, Sigmoid

    class TinyMLP:
        def __init__(self):
            self.l1 = Linear(2, hidden_size)
            self.act = Tanh()
            self.l2 = Linear(hidden_size, 1)
            self.sig = Sigmoid()

        def __call__(self, x):
            h = [self.act(v) for v in self.l1(x)]
            out = self.l2(h)
            return [self.sig(v) for v in out]

        def parameters(self):
            return self.l1.parameters() + self.l2.parameters()

        def zero_grad(self):
            self.l1.zero_grad()
            self.l2.zero_grad()

    return TinyMLP()


def _make_xor_data():
    """Return XOR dataset."""
    from neuralearn.value import Value

    inputs = [
        [Value(0.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(0.0)],
        [Value(1.0), Value(1.0)],
    ]
    targets = [
        [Value(0.0)],
        [Value(1.0)],
        [Value(1.0)],
        [Value(0.0)],
    ]
    return inputs, targets


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a polished end-to-end XOR classification demo."""
    from neuralearn.losses import binary_cross_entropy
    from neuralearn.optimizers import Adam
    from neuralearn.training import Trainer

    epochs = getattr(args, "epochs", 300)
    lr = getattr(args, "lr", 0.05)
    hidden = getattr(args, "hidden", 8)
    do_plot = getattr(args, "plot", False)
    output_dir = getattr(args, "output", None)

    inputs, targets = _make_xor_data()
    model = _build_xor_demo_model(hidden_size=hidden)
    optimizer = Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, binary_cross_entropy, optimizer)

    n_params = len(model.parameters())

    print("=" * 50)
    print("  NeuraLearn Demo — XOR Classification")
    print("=" * 50)
    print()
    print(f"  Dataset:    XOR (4 samples, 2 features)")
    print(f"  Model:      MLP 2 -> {hidden} -> 1 (Tanh, Sigmoid)")
    print(f"  Parameters: {n_params}")
    print(f"  Loss:       Binary Cross-Entropy")
    print(f"  Optimizer:  Adam (lr={lr})")
    print(f"  Epochs:     {epochs}")
    print()

    # Evaluate before training
    pre_preds = []
    for x in inputs:
        pred = model(x)
        pre_preds.append(pred[0].data)
    target_vals = [t[0].data for t in targets]
    pre_loss = sum(
        (p - t) ** 2 for p, t in zip(pre_preds, target_vals)
    ) / len(targets)

    print(f"  Initial MSE: {pre_loss:.6f}")
    print()
    print("  Training...")
    print()

    history = trainer.fit(inputs, targets, epochs=epochs)

    # Print training summary
    losses = history["loss"]
    print(f"    Epoch  1: loss = {losses[0]:.6f}")
    if epochs > 10:
        mid = epochs // 2
        print(f"    Epoch {mid:3d}: loss = {losses[mid - 1]:.6f}")
    print(f"    Epoch {epochs:3d}: loss = {losses[-1]:.6f}")
    print()

    # Final evaluation
    correct = 0
    print("  Predictions:")
    print(f"    {'Input':<16} {'Target':>8} {'Raw':>8} {'Class':>8} {'Correct':>8}")
    print(f"    {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for x, t in zip(inputs, targets):
        pred = model(x)
        raw = pred[0].data
        predicted_class = 1.0 if raw >= 0.5 else 0.0
        is_correct = predicted_class == t[0].data
        if is_correct:
            correct += 1
        inp_str = f"({x[0].data:.0f}, {x[1].data:.0f})"
        print(f"    {inp_str:<16} {t[0].data:>8.1f} {raw:>8.4f} {predicted_class:>8.1f} {'  YES' if is_correct else '   NO':>8}")

    print()
    print(f"  Accuracy:   {correct}/4 ({100 * correct / 4:.0f}%)")
    print(f"  Final loss: {losses[-1]:.6f}")
    print()

    if losses[-1] < 0.01 and correct == 4:
        print("  Result:     SUCCESS — XOR learned perfectly")
    elif correct >= 3:
        print("  Result:     PARTIAL — 3/4 correct")
    else:
        print("  Result:     INCOMPLETE — training may need more epochs")
    print()

    # Optional plot
    if do_plot:
        _generate_demo_plot(losses, inputs, targets, model, output_dir)

    return 0


def _generate_demo_plot(losses, inputs, targets, model, output_dir: str | None):
    """Generate training plot and prediction scatter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not installed — skipping visualization")
        print("         Install with: pip install matplotlib")
        return

    import os

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "neuralearn_demo_output")
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, "b-", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("XOR Training Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(output_dir, "demo_loss_curve.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    # Plot 2: Predictions
    fig, ax = plt.subplots(figsize=(6, 6))
    for x, t in zip(inputs, targets):
        pred = model(x)
        raw = pred[0].data
        color = "green" if (raw >= 0.5) == (t[0].data == 1.0) else "red"
        marker = "o" if t[0].data == 1.0 else "s"
        ax.scatter(x[0].data, x[1].data, c=color, s=200, marker=marker,
                   edgecolors="black", linewidths=1, zorder=5)
        ax.annotate(f"{raw:.3f}", (x[0].data, x[1].data),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_title("XOR Predictions")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pred_path = os.path.join(output_dir, "demo_predictions.png")
    fig.savefig(pred_path, dpi=150)
    plt.close(fig)

    print(f"  Plots saved to:")
    print(f"    {loss_path}")
    print(f"    {pred_path}")
    print()


def cmd_example(args: argparse.Namespace) -> int:
    """Run a tiny XOR training example (brief output)."""
    from neuralearn.losses import binary_cross_entropy
    from neuralearn.optimizers import Adam
    from neuralearn.training import Trainer

    inputs, targets = _make_xor_data()
    model = _build_xor_demo_model(hidden_size=8)
    optimizer = Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model, binary_cross_entropy, optimizer)

    print("Training XOR with a 2-layer MLP (2->8->1, Tanh, Sigmoid)...")
    history = trainer.fit(inputs, targets, epochs=300)
    final_loss = history["loss"][-1]

    correct = 0
    for x, t in zip(inputs, targets):
        pred = model(x)
        predicted_class = 1.0 if pred[0].data >= 0.5 else 0.0
        if predicted_class == t[0].data:
            correct += 1

    print(f"Final loss: {final_loss:.6f}")
    print(f"Accuracy: {correct}/4 ({100 * correct / 4:.0f}%)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="neuralearn",
        description="Neural Network From Scratch — a from-scratch deep learning framework",
    )
    parser.add_argument(
        "--version", action="store_true", default=False,
        help="print version and exit",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("info", help="show project information")

    demo_parser = subparsers.add_parser("demo", help="run end-to-end XOR demo")
    demo_parser.add_argument("--epochs", type=int, default=300, help="training epochs (default: 300)")
    demo_parser.add_argument("--lr", type=float, default=0.05, help="learning rate (default: 0.05)")
    demo_parser.add_argument("--hidden", type=int, default=8, help="hidden layer size (default: 8)")
    demo_parser.add_argument("--plot", action="store_true", help="generate training plots (requires matplotlib)")
    demo_parser.add_argument("--output", type=str, default=None, help="output directory for plots")

    subparsers.add_parser("example", help="run a tiny XOR training example (brief)")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        return cmd_version(args)

    if args.command == "info":
        return cmd_info(args)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "example":
        return cmd_example(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
