# 03 — Project Plan

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 9 COMPLETE

## Overview

This document defines the staged implementation roadmap for the project. Each stage has clear objectives, acceptance criteria, and dependencies.

**Current Stage:** Stage 9 — End-to-End Experiments (complete)

## Stage Overview

```
Stage 0  — Project Definition & Documentation     ✓ COMPLETE
Stage 1  — Core Value / Computational Graph       ✓ COMPLETE
Stage 2  — Extended Autodiff Operations          ✓ COMPLETE
Stage 3  — Gradient Checking                    ✓ COMPLETE
Stage 4  — Parameters and Layers            ✓ COMPLETE
Stage 5  — Loss Functions                    ✓ COMPLETE
Stage 6  — Optimizers                      ✓ COMPLETE
Stage 7  — Training Engine                  ✓ COMPLETE
Stage 8  — Dataset & DataLoader             ✓ COMPLETE
Stage 9  — End-to-End Experiments           ✓ COMPLETE
Stage 10 — Controlled Experiments
Stage 11 — Visualization and Evaluation
Stage 12 — Packaging
Stage 13 — Reproducibility / Demonstration
Stage 14 — Final Documentation and Portfolio Polish
```

---

## Stage 0 — Project Definition & Documentation

**Status:** COMPLETE  
**Dependencies:** None

### Objective

Establish project foundation through documentation and scaffolding.

### Major Work

- Define project requirements and technical design
- Create project structure and documentation
- Establish development methodology

### Expected Learning

- Clear understanding of project scope
- Documentation-first development practice
- Technical writing for ML projects

### Acceptance Criteria

- [x] Repository structure exists
- [x] All required documentation exists
- [x] Documentation is internally consistent
- [x] README accurately describes current status
- [x] No neural-network implementation exists

### Expected Artifacts

- README.md
- 14 documentation files
- Project directory structure
- .gitignore

---

## Stage 1 — Core Value / Computational Graph

**Status:** COMPLETE (approved)
**Dependencies:** Stage 0

### Objective

Build the smallest real, mathematically correct reverse-mode automatic differentiation engine around a scalar `Value` abstraction. Implement the foundational computational graph, forward operations, backward propagation, and gradient accumulation.

### Scope Note

Stage 1 supports **scalar** `Value` objects only. Tensor/vector/matrix support is deferred to later stages.

### Major Work

- Design and implement `Value` class with scalar data, gradient storage, parent tracking, operation metadata, and backward function
- Implement forward operations: addition, multiplication, negation, subtraction, scalar power
- Implement reverse-mode autodiff: local derivative rules, backward propagation, topological ordering, gradient accumulation
- Support `Value`-scalar interoperability (e.g., `x + 2`, `2 * x`, `x ** 2`)
- Comprehensive testing: unit tests, mathematical correctness, gradient accumulation, numerical gradient verification

### Expected Learning

- How computational graphs represent operations as nodes and edges
- Why topological ordering is required for correct reverse-mode autodiff
- How gradient accumulation handles shared nodes in branching graphs
- How chain rule is applied at each node during backward pass

### Acceptance Criteria

- [x] `Value` class stores scalar data, gradient, parent dependencies, operation metadata, backward function
- [x] Addition (`+`) implemented with correct forward and backward
- [x] Multiplication (`*`) implemented with correct forward and backward
- [x] Negation (`-x`) implemented with correct forward and backward
- [x] Subtraction (`-`) implemented with correct forward and backward
- [x] Scalar power (`**`) implemented with correct forward and backward
- [x] Scalar interoperability works for all operations
- [x] `backward()` initializes root gradient to 1.0
- [x] Reverse topological traversal propagates gradients correctly
- [x] Gradient accumulation works for shared/branching nodes
- [x] Numerical gradient checking passes for representative expressions
- [x] All unit tests pass
- [ ] No forbidden autodiff frameworks used
- [ ] Documentation accurately reflects implementation

### Expected Artifacts

- `src/neuralearn/value.py` — Value class with forward ops and backward
- `tests/test_value.py` — Comprehensive tests
- Updated documentation

---

## Stage 2 — Extended Autodiff Operations

**Status:** Stage 2 COMPLETE  
**Dependencies:** Stage 1

### Objective

Extend the scalar autodiff engine with division, reciprocal, exp, log, tanh, and ReLU operations — including forward computation and reverse-mode backward rules. Document domain/edge-case behavior.

### Scope Note

Stage 2 remains **scalar-only** — no tensors, layers, losses, optimizers, or training.

### Major Work

- Implement division (`x / y`), reciprocal (`1 / x`), exp, log, tanh, ReLU operations
- Add corresponding `_backward` closures for each new operation
- Numerical gradient verification for all new operations
- Document domain/edge-case behavior (log domain, ReLU at zero, division by zero)

### Expected Learning

- How division backward uses the quotient rule
- How exp/log/tanh backward rules follow from calculus
- How ReLU's non-smoothness is handled at x=0
- Why log requires domain restrictions (x > 0)

### Acceptance Criteria

- [x] Division (`__truediv__`, `__rtruediv__`) implemented with correct forward and backward
- [x] Exp (`exp()`) implemented with correct forward and backward
- [x] Log (`log()`) implemented with correct forward and backward (domain: x > 0)
- [x] Tanh (`tanh()`) implemented with correct forward and backward
- [x] ReLU (`relu()`) implemented with correct forward and backward
- [x] All operations have backward functions that follow the chain rule
- [x] Numerical gradient checking passes for all new operations
- [x] Scalar interoperability works for all new operations
- [x] Domain/edge-case behavior documented (log domain, ReLU at zero, division by zero)
- [x] ReLU at zero convention explicitly chosen and documented
- [x] Chained/branching/shared graphs work correctly with new operations
- [x] Repeated backward (D19) works correctly with new operations
- [x] All existing Stage 1 tests still pass (zero regressions)
- [x] New tests are added to `tests/test_value.py`
- [x] No forbidden autodiff frameworks used
- [x] Documentation accurately reflects implementation

### Expected Artifacts

- `src/neuralearn/value.py` — Value class extended with new operations
- `tests/test_value.py` — Comprehensive Stage 2 tests added
- Updated documentation

---

## Stage 3 — Gradient Checking

**Status:** COMPLETE  
**Dependencies:** Stage 2

### Objective

Implement a robust, reusable gradient-checking subsystem that independently verifies analytical autodiff gradients against numerical finite-difference approximations.

### Scope Note

Stage 3 remains **scalar-only** — no tensors, layers, losses, optimizers, or training.

### Major Work

- Create `src/neuralearn/gradient_check.py` with reusable numerical gradient utilities
- Central-difference finite-difference approximation with configurable epsilon
- Gradient comparison with absolute and relative tolerance
- Tests covering individual operations, composed graphs, multi-input graphs, deep graphs
- Document gradient-checking methodology, epsilon selection, tolerance, limitations

### Expected Learning

- How central-difference finite differences approximate derivatives
- Why analytical and numerical gradients may differ (truncation error, roundoff)
- How to choose epsilon and tolerance for meaningful gradient checks
- Limitations around nondifferentiable points (ReLU at zero)

### Acceptance Criteria

- [x] `gradient_check.py` module created with `numerical_grad` and `gradient_check` functions
- [x] Numerical gradients use central differences (not autodiff)
- [x] Analytical gradients come from `Value.backward()`
- [x] One-input and multi-input gradient checks work
- [x] All individual Stage 1 operations are covered
- [x] All Stage 2 operations are covered
- [x] Composed graphs are covered
- [x] Deep graphs are covered
- [x] Gradient comparisons use meaningful tolerances
- [x] ReLU zero convention handled/documented
- [x] Numerical edge cases documented
- [x] Existing 152 tests still pass (zero regressions)
- [x] New Stage 3 tests pass
- [x] D19 repeated-backward semantics remain unchanged
- [x] No forbidden frameworks added
- [x] No Stage 4 functionality implemented
- [x] Documentation accurately reflects implementation

### Expected Artifacts

- `src/neuralearn/gradient_check.py` — Reusable gradient-checking module
- `tests/test_gradient_check.py` — Comprehensive gradient-checking tests
- Updated documentation

---

## Stage 4 — Parameters and Layers

**Status:** COMPLETE  
**Dependencies:** Stage 4

### Objective

Implement trainable parameters and neural-network layers.

### Major Work

- Implement Parameter class (subclass of Value)
- Implement Module base class
- Implement Neuron class
- Implement Linear layer
- Implement ReLU activation layer
- Implement Tanh activation layer

### Expected Learning

- How parameters differ from values
- How layers compose operations
- How forward pass builds computational graph

### Acceptance Criteria

- [x] Parameter subclasses Value; IS-A Value, participates in computational graph
- [x] Parameter has `requires_grad` flag (default True)
- [x] Parameter has `zero_grad()` method to reset `.grad` to 0.0
- [x] Module is a lightweight base class (not abstract)
- [x] Module provides `forward()`, `parameters()`, `zero_grad()`
- [x] Neuron computes weighted sum + bias correctly
- [x] Linear layer computes y = Wx + b correctly
- [x] Linear parameters count is nout * (nin + 1)
- [x] ReLU layer wraps existing Value.relu()
- [x] Tanh layer wraps existing Value.tanh()
- [x] All parameters are discoverable via Module.parameters()
- [x] Unit tests pass for all layers

### Expected Artifacts

- `src/neuralearn/parameter.py`
- `src/neuralearn/layers.py`
- `tests/test_parameter.py`
- `tests/test_layers.py`

---

## Stage 5 — Loss Functions

**Status:** IN PROGRESS  
**Dependencies:** Stage 4

### Objective

Implement loss functions for training.

### Major Work

- Implement Mean Squared Error loss
- Implement Binary Cross-Entropy loss
- Verify gradients through loss functions

### Expected Learning

- How loss functions connect predictions to gradients
- Why different losses suit different problems
- How loss gradients affect training

### Acceptance Criteria

- [ ] MSE loss computes correctly
- [ ] Binary Cross-Entropy loss computes correctly
- [ ] Gradients through losses are correct
- [ ] Unit tests pass for all losses

### Expected Artifacts

- `src/neuralearn/losses.py`
- `tests/test_losses.py`

---

## Stage 6 — Optimizers

**Status:** PLANNED  
**Dependencies:** Stage 4

### Objective

Implement optimizers that update parameters based on gradients.

### Major Work

- Implement SGD optimizer
- Implement Momentum optimizer
- Implement Adam optimizer
- Verify parameter updates

### Expected Learning

- How different optimizers modify gradient descent
- Why momentum helps with convergence
- How adaptive learning rates work

### Acceptance Criteria

- [ ] SGD updates parameters correctly
- [ ] Momentum tracks velocity correctly
- [ ] Adam tracks first and second moments correctly
- [ ] Unit tests pass for all optimizers

### Expected Artifacts

- `src/neuralearn/optimizers.py`
- `tests/test_optimizers.py`

---

## Stage 7 — Training Engine

**Status:** IN PROGRESS  
**Dependencies:** Stages 4, 5, 6

### Objective

Implement a reusable training engine that connects model, loss, and optimizer into a complete training lifecycle.

### Major Work

- Implement Trainer class with fit() and evaluate() methods
- Implement training lifecycle: forward → loss → backward → step → zero_grad
- Support configurable epochs with validation
- Return training history with per-epoch loss values
- Provide evaluation method that does not modify parameters
- Validate inputs and provide clear error messages

### Expected Learning

- How training loop orchestrates model, loss, and optimizer components
- Why gradient reset is necessary between samples/epochs
- How training history enables monitoring and debugging
- The distinction between training (parameter-updating) and evaluation (inference-only)

### Acceptance Criteria

- [x] Trainer connects model, loss_fn, optimizer
- [x] Training loop completes forward/backward/update/zero_grad cycle
- [x] Configurable epochs with validation
- [x] Loss history returned as dict with per-epoch mean loss
- [x] Evaluation computes loss without modifying parameters
- [x] Deterministic training on fixed data
- [x] Simple learning experiment demonstrates loss reduction
- [x] Code is modular and testable

### Expected Artifacts

- `src/neuralearn/training.py`
- `tests/test_training.py`

---

## Stage 8 — Dataset & DataLoader

**Status:** IN PROGRESS  
**Dependencies:** Stage 7

### Objective

Implement lightweight Dataset and DataLoader abstractions for structured data iteration with batching and shuffling support.

### Major Work

- Implement Dataset class for paired input/target storage
- Implement DataLoader class with batching, shuffling, drop_last
- Integrate DataLoader with Trainer.fit()
- Deterministic shuffling with explicit seed
- Preserve per-sample training semantics (D39)

### Expected Learning

- How data loading separates storage from iteration
- How batching affects training iteration patterns
- How deterministic shuffling enables reproducible experiments
- How DataLoader integrates with existing training lifecycle

### Acceptance Criteria

- [x] Dataset stores and validates paired inputs/targets
- [x] DataLoader yields correct batches
- [x] Shuffle preserves input-target pairing
- [x] Seed produces deterministic ordering
- [x] drop_last handles partial batches
- [x] len() returns correct batch count
- [x] Trainer accepts DataLoader in fit()
- [x] Existing training behavior preserved

### Expected Artifacts

- `src/neuralearn/datasets.py`
- `src/neuralearn/dataloaders.py`
- `tests/test_dataset.py`
- `tests/test_dataloader.py`

---

## Stage 9 — XOR Experiment

**Status:** PLANNED  
**Dependencies:** Stage 8

### Objective

Demonstrate training on XOR classification problem.

### Major Work

- Define XOR dataset
- Configure model architecture
- Train model
- Visualize decision boundary
- Document results

### Expected Learning

- How neural networks learn nonlinear functions
- Why hidden layers are necessary for XOR
- How training dynamics affect convergence

### Acceptance Criteria

- [ ] Model trains on XOR problem
- [ ] Training loss decreases
- [ ] Decision boundary is visualized
- [ ] Results are documented

### Expected Artifacts

- `experiments/xor_experiment.py`
- `experiments/results/xor_results.md`
- Experiment logs

---

## Stage 10 — Nonlinear Regression

**Status:** PLANNED  
**Dependencies:** Stage 8

### Objective

Demonstrate training on nonlinear regression problem.

### Major Work

- Define regression dataset
- Configure model architecture
- Train model
- Visualize predictions
- Document results

### Expected Learning

- How neural networks fit complex functions
- How regression differs from classification
- How loss function choice affects training

### Acceptance Criteria

- [ ] Model trains on regression problem
- [ ] Training loss decreases
- [ ] Predictions are visualized
- [ ] Results are documented

### Expected Artifacts

- `experiments/regression_experiment.py`
- `experiments/results/regression_results.md`
- Experiment logs

---

## Stage 10 — Controlled Experiments

**Status:** PLANNED  
**Dependencies:** Stages 8, 9

### Objective

Systematically investigate factors affecting training.

### Major Work

- Design controlled experiment methodology
- Compare optimizers (SGD vs Momentum vs Adam)
- Study learning rate effects
- Study architecture effects (hidden size, depth, activation)
- Document findings

### Expected Learning

- How different factors affect training
- Why controlled experiments require systematic methodology
- How to design fair comparisons

### Acceptance Criteria

- [ ] Experiment methodology is documented
- [ ] Optimizer comparison is completed
- [ ] Learning rate study is completed
- [ ] Architecture study is completed
- [ ] Results are documented with methodology

### Expected Artifacts

- `experiments/controlled/`
- `experiments/results/controlled_results.md`
- Experiment logs

---

## Stage 11 — Visualization and Evaluation

**Status:** PLANNED  
**Dependencies:** Stages 8, 9, 10

### Objective

Create comprehensive visualizations of training and results.

### Major Work

- Implement training loss visualization
- Implement prediction visualization
- Implement decision boundary visualization
- Implement experiment comparison plots

### Expected Learning

- How visualization aids understanding
- What metrics are most informative
- How to present results clearly

### Acceptance Criteria

- [ ] Training curves are plotted
- [ ] Regression predictions are visualized
- [ ] Decision boundaries are visualized
- [ ] Experiment comparisons are visualized

### Expected Artifacts

- `src/neuralearn/visualization.py`
- Visualization scripts
- Result images

---

## Stage 12 — Packaging

**Status:** PLANNED  
**Dependencies:** All implementation stages

### Objective

Package the project for distribution and reuse.

### Major Work

- Create setup.py or pyproject.toml
- Create requirements.txt
- Create installation documentation
- Create example scripts

### Expected Learning

- How to package Python projects
- How to document installation
- How to create reusable examples

### Acceptance Criteria

- [ ] Project can be installed via pip
- [ ] Requirements are documented
- [ ] Installation instructions are clear
- [ ] Examples work correctly

### Expected Artifacts

- `setup.py` or `pyproject.toml`
- `requirements.txt`
- Installation documentation
- Example scripts

---

## Stage 13 — Reproducibility / Demonstration

**Status:** PLANNED  
**Dependencies:** Stage 12

### Objective

Ensure full reproducibility and create demonstration materials.

### Major Work

- Verify reproducible results
- Create demonstration scripts
- Document reproduction steps
- Test reproduction from clean environment

### Expected Learning

- How to ensure reproducibility
- How to document reproduction
- How to create effective demonstrations

### Acceptance Criteria

- [ ] Results are reproducible from documentation
- [ ] Demonstration scripts work correctly
- [ ] Reproduction steps are clear
- [ ] Clean environment reproduction is verified

### Expected Artifacts

- Reproduction documentation
- Demonstration scripts
- Verified reproducible results

---

## Stage 14 — Final Documentation and Portfolio Polish

**Status:** PLANNED  
**Dependencies:** Stage 13

### Objective

Finalize documentation and prepare for portfolio presentation.

### Major Work

- Review all documentation for consistency
- Create portfolio-quality README
- Ensure all experiments are documented
- Final code review

### Expected Learning

- How to present technical work professionally
- How to write portfolio-quality documentation
- How to demonstrate engineering methodology

### Acceptance Criteria

- [ ] All documentation is consistent and accurate
- [ ] README presents project professionally
- [ ] All experiments are documented
- [ ] Code meets quality standards

### Expected Artifacts

- Updated README.md
- Updated all documentation
- Final code review

---

## Dependencies Summary

```
Stage 0 → Stage 1 → Stage 2 → Stage 3
                     ↓
                    Stage 4 → Stage 6 → Stage 7 → Stage 8
                     ↓         ↓         ↓         ↓
                    Stage 5   Stage 6   Stage 9 → Stage 10
                                             ↓
                                            Stage 11
                                             ↓
                                            Stage 12 → Stage 13 → Stage 14
```

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Gradient checking fails | High | Implement early, verify incrementally |
| Training doesn't converge | Medium | Start with simple problems, verify each component |
| Scope creep | Medium | Strict adherence to documented requirements |
| Documentation drift | Low | Update documentation with each stage |
