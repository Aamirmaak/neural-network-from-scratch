# Project 001 — Neural Network From Scratch

**Status:** Stage 3 COMPLETE

## Overview

An educational deep-learning framework built from first principles to demonstrate understanding of the fundamental machinery behind neural-network training.

This project implements a small but complete neural-network training system without using any existing deep-learning frameworks or automatic-differentiation libraries. The goal is to build strong AI/ML engineering fundamentals through real implementation work.

**Current State:** Stage 3 gradient-checking module implemented and validated. Reusable `gradient_check.py` with numerical gradient verification. 195 tests passing. Ready for git checkpoint.

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

## Intended Final System

The completed project will be a small deep-learning framework supporting:

- Tensor/Value abstraction with gradient tracking
- Reverse-mode automatic differentiation
- Neural-network layers (Linear, ReLU, Tanh)
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
| Parameters | PLANNED | Trainable model parameters |
| Layers | PLANNED | Linear, ReLU, Tanh layers |
| Losses | PLANNED | MSE, Binary Cross-Entropy |
| Optimizers | PLANNED | SGD, Momentum, Adam |
| Training Loop | PLANNED | Forward, backward, update cycle |
| Experiments | PLANNED | XOR, regression, controlled comparisons |
| Visualization | PLANNED | Training curves, decision boundaries |

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
| XOR Classification | Learn nonlinear decision boundary | PLANNED |
| Nonlinear Regression | Fit complex function | PLANNED |
| Optimizer Comparison | SGD vs Momentum vs Adam | PLANNED |
| Learning Rate Study | Effect on convergence | PLANNED |
| Architecture Search | Hidden size, depth, activation | PLANNED |

See [06_EXPERIMENT_PLAN.md](06_EXPERIMENT_PLAN.md) for full experiment details.

## Testing Philosophy

Correctness is established through:

1. **Unit tests** for all arithmetic and gradient operations
2. **Numerical gradient checking** via finite differences
3. **Integration tests** for training flows
4. **Regression tests** to prevent silent breakage

A demo running successfully is NOT sufficient evidence of correctness.

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
│       └── gradient_check.py
├── tests/
├── experiments/
├── configs/
└── scripts/
```

## Technology Constraints

- **Language:** Python
- **Numerical:** NumPy allowed
- **Forbidden:** PyTorch, TensorFlow, JAX, Keras, automatic-differentiation libraries
- **Purpose:** Implement learning machinery ourselves

## Development Methodology

- Documentation-first approach
- Staged implementation with validation at each stage
- Correctness before performance
- Modular design with clear interfaces
- Comprehensive testing

## Current Milestone

**Stage 3 — Gradient Checking**

Stage 3 validates the autodiff engine with numerical gradient checking:
- Reusable `gradient_check.py` module with `numerical_grad` and `gradient_check`
- Central-difference finite-difference approximation (ε=1e-5)
- Gradient comparison with absolute and relative tolerance
- Tests covering individual operations, composed graphs, multi-input, deep graphs
- 195 passing tests (152 existing + 43 new)

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
