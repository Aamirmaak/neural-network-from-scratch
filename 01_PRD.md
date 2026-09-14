# 01 — Product Requirements Document

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 0 — Planning

## Problem Statement

Modern deep-learning frameworks (PyTorch, TensorFlow, JAX) provide powerful abstractions that enable rapid development but obscure the fundamental mechanics of neural-network training. Engineers using these frameworks may lack deep understanding of:

- How gradients are computed through automatic differentiation
- Why computational graphs must be constructed in specific ways
- What optimizers actually modify during training
- How architectural choices affect learning dynamics

Building a neural-network framework from scratch forces complete understanding of every component in the training pipeline. This understanding is critical for:

1. Debugging training failures in production systems
2. Making informed architectural decisions
3. Contributing to ML research
4. Designing novel training approaches

## Goals

### Primary Goals

1. **Educational:** Implement all core components of a neural-network training system from first principles
2. **Correctness:** Produce a system where every gradient can be verified mathematically
3. **Understanding:** Demonstrate mastery of the machinery behind modern deep learning
4. **Portfolio:** Create a technical artifact that demonstrates AI/ML engineering ability

### Success Criteria

- Complete reverse-mode automatic differentiation system
- Functional neural-network training on simple problems
- Numerical gradient checking validates all implemented gradients
- Experiments demonstrate controlled investigation of training behavior
- Documentation clearly explains design decisions and learning

## Non-Goals

The following are explicitly outside the initial scope:

1. **Production deployment:** This is an educational project, not a framework for production use
2. **GPU acceleration:** All computation runs on CPU
3. **Distributed training:** Single-device only
4. **Convolutional layers:** Focus on fully-connected layers only
5. **Recurrent networks:** No RNN/LSTM/Transformer support
6. **Advanced regularization:** Dropout, batch normalization are future considerations
7. **Model serialization:** No save/load functionality in initial scope
8. **Performance optimization:** Correctness before speed
9. **Web interfaces:** No UI unless it provides genuine educational value

## Target User

### Primary User

The project author, using the implementation as a learning and research-engineering exercise.

### Secondary User

Technical portfolio reviewers (recruiters, hiring managers, other engineers) who will evaluate:

1. Code quality and organization
2. Depth of technical understanding
3. Engineering methodology
4. Documentation quality
5. Testing approach

## Functional Requirements

### FR-01: Tensor / Value Abstraction

**Priority:** Critical  
**Stage:** 1 — Core Value / Tensor Abstraction

The numerical abstraction must eventually support:

| Feature | Description | Status |
|---------|-------------|--------|
| Value storage | Hold numerical data | PLANNED |
| Gradient storage | Hold gradient with respect to loss | PLANNED |
| Parent tracking | Track which operations produced this value | PLANNED |
| Operation info | Record the operation that created this value | PLANNED |
| Backward function | Reference to the function that computes gradients for this operation | PLANNED |

### FR-02: Reverse-Mode Automatic Differentiation

**Priority:** Critical  
**Stage:** 2 — Automatic Differentiation

The autodiff system must eventually support:

| Operation | Forward | Backward |
|-----------|---------|----------|
| Addition | a + b | da = dout, db = dout |
| Subtraction | a - b | da = dout, db = -dout |
| Multiplication | a * b | da = b * dout, db = a * dout |
| Division | a / b | da = dout / b, db = -a * dout / b² |
| Power | a^n | da = n * a^(n-1) * dout |
| Negation | -a | da = -dout |
| Exponential | exp(a) | da = exp(a) * dout |
| Logarithm | log(a) | da = dout / a |
| Tanh | tanh(a) | da = (1 - tanh(a)²) * dout |
| ReLU | max(0, a) | da = dout if a > 0, else 0 |

The backward pass must use topological ordering to ensure correct gradient computation.

### FR-03: Neural-Network Components

**Priority:** High  
**Stage:** 4 — Parameters and Layers

| Component | Description | Status |
|-----------|-------------|--------|
| Parameter | Trainable weight with gradient | PLANNED |
| Linear | Fully-connected layer (Wx + b) | PLANNED |
| ReLU | Rectified linear unit activation | PLANNED |
| Tanh | Hyperbolic tangent activation | PLANNED |
| Sequential | Container for stacking layers | PLANNED |

### FR-04: Loss Functions

**Priority:** High  
**Stage:** 5 — Loss Functions

| Loss | Use Case | Status |
|------|----------|--------|
| Mean Squared Error | Regression | PLANNED |
| Binary Cross-Entropy | Binary classification | PLANNED |

### FR-05: Optimizers

**Priority:** High  
**Stage:** 6 — Optimizers

| Optimizer | Description | Status |
|-----------|-------------|--------|
| SGD | Basic stochastic gradient descent | PLANNED |
| Momentum | SGD with momentum | PLANNED |
| Adam | Adaptive moment estimation | PLANNED |

### FR-06: Training Infrastructure

**Priority:** High  
**Stage:** 7 — Training Infrastructure

The training loop must eventually support:

- Forward pass through model
- Loss computation
- Backward pass (gradient computation)
- Optimizer step (parameter update)
- Gradient reset
- Metrics computation
- Logging
- Checkpoints where appropriate

### FR-07: Experiments

**Priority:** Medium  
**Stage:** 8-10 — Experiments

At minimum, the project must demonstrate:

| Experiment | Purpose |
|------------|---------|
| XOR Classification | Learn nonlinear decision boundary |
| Nonlinear Regression | Fit complex function |
| Optimizer Comparison | Controlled comparison of SGD, Momentum, Adam |
| Learning Rate Study | Effect on convergence |
| Architecture Search | Effect of hidden size, depth, activation |

### FR-08: Gradient Checking

**Priority:** Critical  
**Stage:** 3 — Gradient Checking

Numerical finite-difference gradient checking is a critical requirement. Every implemented gradient must be verifiable against numerical approximation.

### FR-09: Testing

**Priority:** High  
**Stage:** 0 (Strategy) → 3+ (Implementation)

Automated tests must eventually cover:

| Test Category | Coverage |
|---------------|----------|
| Arithmetic operations | All operations in FR-02 |
| Graph construction | Forward and backward connections |
| Gradients | All gradients match numerical approximation |
| Layers | Forward pass, parameter shapes |
| Losses | Forward computation, gradient flow |
| Optimizers | Parameter updates |

### FR-10: Visualization

**Priority:** Medium  
**Stage:** 11 — Visualization and Evaluation

Eventually produce:

- Training loss curves
- Validation/test loss curves where appropriate
- Regression predictions vs actual
- XOR decision boundary visualization
- Experiment comparison plots

### FR-11: Packaging and Usability

**Priority:** Medium  
**Stage:** 12-13 — Packaging and Reproducibility

Eventually provide:

- Reproducible installation instructions
- Reproducible environment setup
- Clean CLI or demonstration entry point
- Example usage
- Documented environment/setup
- Reproducible results

## Dependencies

| Dependency | Purpose | Required |
|------------|---------|----------|
| Python | Implementation language | Yes |
| NumPy | Numerical operations | Yes |
| Matplotlib | Visualization | Optional |
| pytest | Testing | Optional |

## Constraints

1. No deep-learning frameworks in core implementation
2. No automatic-differentiation libraries in core implementation
3. All core components must be implemented from first principles
4. Documentation must clearly distinguish PLANNED from IMPLEMENTED features
5. No fabricated results or benchmarks

## Open Questions

None at this stage.

## Appendix: Requirements Traceability

| Requirement | Stage | Dependencies |
|-------------|-------|--------------|
| FR-01: Value/Tensor | 1 | None |
| FR-02: Autodiff | 2 | FR-01 |
| FR-03: Gradient Checking | 3 | FR-02 |
| FR-04: Parameters/Layers | 4 | FR-02 |
| FR-05: Loss Functions | 5 | FR-02 |
| FR-06: Optimizers | 6 | FR-04 |
| FR-07: Training Infrastructure | 7 | FR-04, FR-05, FR-06 |
| FR-08: Experiments | 8-10 | FR-07 |
| FR-09: Testing | 0+ | All |
| FR-10: Visualization | 11 | FR-07, FR-08 |
| FR-11: Packaging | 12-13 | All |
