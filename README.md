# NeuraLearn — Neural Network From Scratch

**Version:** 1.4.0 | **Status:** Open-Source Release | **License:** MIT

An educational neural-network framework built from first principles. Implements reverse-mode automatic differentiation, layers, losses, optimizers, and training infrastructure without any deep-learning frameworks.

## Why This Exists

Modern deep-learning frameworks abstract away the mechanics of neural-network training. This project builds those mechanics from scratch to demonstrate genuine understanding of:

- Reverse-mode automatic differentiation
- Computational graphs and gradient flow
- Backpropagation and gradient accumulation
- Optimizer behavior (SGD, Momentum, Adam)
- Training loop lifecycle

## What's Included

| Component | Status |
|-----------|--------|
| Scalar autodiff (Value) | Implemented |
| Gradient checking | Implemented |
| Parameters & Modules | Implemented |
| Layers (Linear, ReLU, Tanh, Sigmoid) | Implemented |
| Losses (MSE, Binary Cross-Entropy) | Implemented |
| Optimizers (SGD, Momentum, Adam) | Implemented |
| Training engine (Trainer) | Implemented |
| Dataset & DataLoader | Implemented |
| XOR classification experiment | Implemented |
| Nonlinear regression experiment | Implemented |
| Visualization (loss curves, predictions) | Implemented |
| CLI with demo command | Implemented |
| 521 passing tests | Implemented |

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

For visualization support:

```bash
pip install -e ".[viz]"
```

**Zero external dependencies** for core functionality.

## Quick Start

### Python API

```python
from neuralearn import (
    Value, Linear, Tanh, Sigmoid,
    mse_loss, binary_cross_entropy,
    Adam, Trainer, Dataset, DataLoader,
)

# Define a model
class MLP:
    def __init__(self):
        self.layer1 = Linear(2, 8)
        self.act1 = Tanh()
        self.layer2 = Linear(8, 1)
        self.act2 = Sigmoid()

    def __call__(self, x):
        h = [self.act1(v) for v in self.layer1(x)]
        out = self.layer2(h)
        return [self.act2(v) for v in out]

    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters()

    def zero_grad(self):
        self.layer1.zero_grad()
        self.layer2.zero_grad()

# XOR dataset
inputs = [[Value(0.0), Value(0.0)], [Value(0.0), Value(1.0)],
          [Value(1.0), Value(0.0)], [Value(1.0), Value(1.0)]]
targets = [[Value(0.0)], [Value(1.0)], [Value(1.0)], [Value(0.0)]]

# Train
model = MLP()
optimizer = Adam(model.parameters(), lr=0.01)
trainer = Trainer(model, binary_cross_entropy, optimizer)
history = trainer.fit(inputs, targets, epochs=500)

print(f"Final loss: {history['loss'][-1]:.4f}")
```

### CLI Demo

```bash
neuralearn demo              # run XOR demo
neuralearn demo --plot       # with visualization
neuralearn --version         # print version
neuralearn --help            # usage info
```

## Architecture

```
Value (scalar with gradient tracking)
  → Computational Graph
    → Reverse-Mode Autodiff
      → Parameters (trainable)
        → Layers (Linear, ReLU, Tanh, Sigmoid)
          → Model (composed layers)
            → Loss (MSE, BCE)
              → Optimizer (SGD, Momentum, Adam)
                → Training Loop (Trainer)
                  → Experiments & Visualization
```

## Testing

```bash
python -m pytest tests/ -q    # 521 tests
```

Tests cover: autodiff correctness, gradient checking, layers, losses, optimizers, training, data loading, visualization, API smoke tests, CLI, and demo.

## Project Structure

```
neuralearn/
├── src/neuralearn/          # framework source
│   ├── value.py             # scalar autodiff
│   ├── parameter.py         # trainable values
│   ├── layers.py            # Linear, ReLU, Tanh, Sigmoid
│   ├── losses.py            # MSE, BCE
│   ├── optimizers.py        # SGD, Momentum, Adam
│   ├── training.py          # Trainer
│   ├── datasets.py          # Dataset
│   ├── dataloaders.py       # DataLoader
│   ├── gradient_check.py    # numerical validation
│   ├── visualization.py     # plotting (optional)
│   └── cli.py               # command-line interface
├── tests/                   # 521 tests
├── experiments/             # XOR, regression experiments
├── artifacts/plots/         # generated visualizations
├── pyproject.toml           # package configuration
├── LICENSE                  # MIT license
├── CONTRIBUTING.md          # contribution guide
├── CODE_OF_CONDUCT.md       # community standards
└── SECURITY.md              # security policy
```

## Design Principles

- **Correctness first:** Every operation verified via numerical gradient checking
- **No magic:** All training mechanics visible in source code
- **Testable:** 521 tests including edge cases and integration
- **Minimal dependencies:** Pure Python core, optional matplotlib
- **Reproducible:** Deterministic demos, documented experiments

## Limitations

- Educational scope — not intended for production use
- Scalar values only (no tensors)
- Dense/fully-connected layers only
- Single-device training
- No GPU acceleration

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.
