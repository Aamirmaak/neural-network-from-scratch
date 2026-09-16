# Project 001 — Neural Network From Scratch

**Status:** Stage 12 Complete

## Overview

An educational deep-learning framework built from first principles to demonstrate understanding of the fundamental machinery behind neural-network training.

This project implements a small but complete neural-network training system without using any existing deep-learning frameworks or automatic-differentiation libraries. The goal is to build strong AI/ML engineering fundamentals through real implementation work.

**Current State:** Stage 12 Package Distribution + CLI complete. 507 tests passing. Installable via pip, CLI with `--version`, `info`, and `example` commands.

## Problem / Motivation

Modern deep-learning frameworks abstract away the core mechanics of neural-network training. While this enables rapid development, it can leave engineers without deep understanding of:

- How automatic differentiation works
- Why backpropagation requires specific ordering
- How gradients flow through computational graphs
- What optimizers actually do during training
- Why certain architectural choices matter

Building from scratch forces confrontation with every detail of the training machinery.

## Learning Objectives

1. Implement reverse-mode automatic differentiation from first principles
2. Understand computational graphs and gradient flow
3. Build neural-network components (layers, activations, losses, optimizers)
4. Design reproducible experiments that demonstrate training behavior
5. Develop engineering practices for correctness, modularity, and testability

## Installation

```bash
# Standard install
pip install .

# Development install (editable)
pip install -e .

# With visualization support
pip install -e ".[viz]"

# Full development (pytest + matplotlib)
pip install -e ".[dev]"
```

No external dependencies required for core functionality.

## Quick Start

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

# Create dataset
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

## CLI

After installation, the `neuralearn` command is available:

```bash
neuralearn --version    # print version
neuralearn info         # project information
neuralearn example      # run XOR training demo
neuralearn --help       # usage help
```

## Intended Final System

The completed project will be a small deep-learning framework supporting:

- Tensor/Value abstraction with gradient tracking
- Reverse-mode automatic differentiation
- Neural-network layers (Linear, ReLU, Tanh, Sigmoid)
- Loss functions (MSE, Binary Cross-Entropy)
- Optimizers (SGD, Momentum, Adam)
- Training infrastructure with logging and visualization
- Reproducible experiments (XOR classification, nonlinear regression)

## Planned Technical Components

| Component | Status | Description |
|-----------|--------|-------------|
| Value (scalar) | IMPLEMENTED | Scalar numerical abstraction with gradient tracking |
| Autodiff (basic) | IMPLEMENTED | Reverse-mode autodiff: +, *, neg, -, ** |
| Autodiff (extended) | IMPLEMENTED | Division, exp, log, tanh, ReLU, reciprocal |
| Parameters | IMPLEMENTED | Trainable model parameters |
| Layers | IMPLEMENTED | Linear, ReLU, Tanh layers |
| Losses | IMPLEMENTED | MSE, Binary Cross-Entropy |
| Optimizers | IMPLEMENTED | SGD, Momentum, Adam |
| Training Engine | IMPLEMENTED | Trainer with fit/evaluate lifecycle |
| Dataset | IMPLEMENTED | Paired input/target storage with validation |
| DataLoader | IMPLEMENTED | Batching, shuffling, seed, drop_last |
| Sigmoid | IMPLEMENTED | Sigmoid activation layer |
| Experiments | IMPLEMENTED | XOR classification, nonlinear regression |
| Visualization | IMPLEMENTED | Loss curves, regression fit, classification scatter, comparison plots |

## Architecture Overview

```
Value / Tensor
      ↓
Computational Graph
      ↓
Autodiff
      ↓
Parameters
      ↓
Layers
      ↓
Model
      ↓
Loss
      ↓
Optimizer
      ↓
Training Loop
      ↓
Experiments
```

See [04_ARCHITECTURE.md](04_ARCHITECTURE.md) for detailed architecture.

## Planned Experiments

| Experiment | Objective | Status |
|------------|-----------|--------|
| XOR Classification | Learn nonlinear decision boundary | COMPLETE |
| Nonlinear Regression | Fit complex function | COMPLETE |
| Optimizer Comparison | SGD vs Momentum vs Adam | PLANNED |
| Learning Rate Study | Effect on convergence | PLANNED |
| Architecture Search | Hidden size, depth, activation | PLANNED |

See [06_EXPERIMENT_PLAN.md](06_EXPERIMENT_PLAN.md) for full experiment details.

## Experiments & Visualization

### Running Experiments

```bash
# XOR Classification
python experiments/experiment_a_xor.py

# Nonlinear Regression
python experiments/experiment_b_regression.py
```

### Generating Plots

```bash
# Requires matplotlib: pip install matplotlib
python experiments/generate_plots.py
```

### Generated Plots

Plots are saved to `artifacts/plots/`:

| Plot | Description |
|------|-------------|
| `xor_loss_curve.png` | XOR training loss over epochs |
| `xor_accuracy_curve.png` | XOR training accuracy over epochs |
| `xor_predictions.png` | MLP prediction scatter for XOR |
| `regression_train_test_loss.png` | Train vs test MSE for regression |
| `regression_fit.png` | MLP fit vs target function |

### Visualization API

```python
from neuralearn.visualization import plot_loss, plot_regression_fit

# Plot training loss
plot_loss(history, title="My Experiment", save_path="loss.png")

# Plot regression fit
plot_regression_fit(x_train, y_train, x_test, y_test, preds, target_fn=fn)
```

matplotlib is an optional dependency — core framework works without it.

## Testing

```bash
# Run full test suite
python -m pytest tests/ -q

# Run specific test file
python -m pytest tests/test_cli.py -v
```

507 tests covering all components: autodiff, layers, losses, optimizers, training, data, visualization, API smoke tests, and CLI.

## Reproducibility

The project will eventually support:

1. Clone repository
2. Create environment
3. Install dependencies
4. Run tests
5. Run experiments
6. Reproduce documented results

## Repository Structure

```
neural-network-from-scratch/
├── README.md
├── pyproject.toml
├── requirements.txt
├── 01_PRD.md
├── 02_TRD.md
├── 03_PROJECT_PLAN.md
├── 04_ARCHITECTURE.md
├── 05_RESEARCH_PLAN.md
├── 06_EXPERIMENT_PLAN.md
├── 07_TESTING_STRATEGY.md
├── 08_DEPLOYMENT_PLAN.md
├── 09_SECURITY.md
├── 10_DECISIONS.md
├── 11_PROGRESS_LOG.md
├── 12_EXPERIMENT_LOG.md
├── 13_LEARNINGS.md
├── 14_CHANGELOG.md
├── src/
│   └── neuralearn/
│       ├── __init__.py
│       ├── value.py
│       ├── gradient_check.py
│       ├── parameter.py
│       ├── layers.py
│       ├── losses.py
│       ├── optimizers.py
│       ├── training.py
│       ├── datasets.py
│       ├── dataloaders.py
│       ├── visualization.py
│       └── cli.py
├── tests/
├── experiments/
├── artifacts/
│   └── plots/
├── configs/
└── scripts/
```

## Technology Constraints

- **Language:** Python (>=3.10)
- **Core dependencies:** None (pure Python)
- **Optional:** matplotlib (for visualization)
- **Forbidden:** PyTorch, TensorFlow, JAX, Keras, automatic-differentiation libraries
- **Purpose:** Implement learning machinery ourselves

## Development Methodology

- Documentation-first approach
- Staged implementation with validation at each stage
- Correctness before performance
- Modular design with clear interfaces
- Comprehensive testing

## Public API

```python
# Core
from neuralearn import Value, Parameter

# Layers
from neuralearn import Module, Neuron, Linear, ReLU, Tanh, Sigmoid

# Losses
from neuralearn import mse_loss, binary_cross_entropy

# Optimizers
from neuralearn import SGD, MomentumSGD, Adam

# Training
from neuralearn import Trainer

# Data
from neuralearn import Dataset, DataLoader

# Validation
from neuralearn import numerical_grad, gradient_check

# Visualization (requires matplotlib)
from neuralearn import (
    plot_loss, plot_train_test_curve, plot_regression_fit,
    plot_xor_predictions, plot_comparison, plot_accuracy,
)

# CLI
import subprocess
subprocess.run(["neuralearn", "--version"])
```

## Current Milestone

**Stage 12 — Package Distribution + CLI**

Stage 12 turns the project into a proper installable Python package:
- `pyproject.toml` with correct `setuptools.build_meta` backend
- `pip install .` works from clean environment
- `pip install -e .` works for development
- CLI entry point: `neuralearn --version`, `neuralearn info`, `neuralearn example`
- 12 new CLI tests (parser + subprocess)
- 507 total tests passing
- Clean build artifacts (sdist + wheel)
- No new ML capabilities added

## Future Roadmap

See [03_PROJECT_PLAN.md](03_PROJECT_PLAN.md) for the full 14-stage roadmap.

## Limitations

- Educational scope: not intended for production use
- Single-device training only
- Limited to dense/fully-connected layers
- No GPU acceleration planned
- No distributed training

## Learning Goals

This project demonstrates:

1. Understanding of automatic differentiation fundamentals
2. Ability to implement ML systems from first principles
3. Engineering practices for correctness and reproducibility
4. Research-engineering methodology
5. Technical communication through documentation
