"""
Lightweight command-line interface for neuralearn.

Usage:
    neuralearn --version
    neuralearn --help
    neuralearn info
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


def cmd_example(args: argparse.Namespace) -> int:
    """Run a tiny XOR classification example."""
    from neuralearn.value import Value
    from neuralearn.layers import Linear, Tanh, Sigmoid
    from neuralearn.losses import binary_cross_entropy
    from neuralearn.optimizers import Adam
    from neuralearn.training import Trainer

    class TinyMLP:
        def __init__(self):
            self.l1 = Linear(2, 8)
            self.act = Tanh()
            self.l2 = Linear(8, 1)
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

    model = TinyMLP()
    optimizer = Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model, binary_cross_entropy, optimizer)

    print("Training XOR with a 2-layer MLP (2->8->1, Tanh, Sigmoid)...")
    history = trainer.fit(inputs, targets, epochs=300)
    final_loss = history["loss"][-1]

    # Verify predictions
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
    subparsers.add_parser("example", help="run a tiny XOR training example")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        return cmd_version(args)

    if args.command == "info":
        return cmd_info(args)
    if args.command == "example":
        return cmd_example(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
