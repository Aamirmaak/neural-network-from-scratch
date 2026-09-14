# 03 — Project Plan

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 2 In Progress

## Overview

This document defines the staged implementation roadmap for the project. Each stage has clear objectives, acceptance criteria, and dependencies.

**Current Stage:** Stage 2 — Extended Autodiff Operations (in progress)

## Stage Overview

```
Stage 0  — Project Definition & Documentation     ✓ COMPLETE
Stage 1  — Core Value / Computational Graph       ✓ COMPLETE
Stage 2  — Extended Autodiff Operations           ◉ CURRENT
Stage 3  — Gradient Checking
Stage 4  — Parameters and Layers
Stage 5  — Loss Functions
Stage 6  — Optimizers
Stage 7  — Training Infrastructure
Stage 8  — XOR Experiment
Stage 9  — Nonlinear Regression
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

- [ ] Repository structure exists
- [ ] All required documentation exists
- [ ] Documentation is internally consistent
- [ ] README accurately describes current status
- [ ] No neural-network implementation exists

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

- [ ] `Value` class stores scalar data, gradient, parent dependencies, operation metadata, backward function
- [ ] Addition (`+`) implemented with correct forward and backward
- [ ] Multiplication (`*`) implemented with correct forward and backward
- [ ] Negation (`-x`) implemented with correct forward and backward
- [ ] Subtraction (`-`) implemented with correct forward and backward
- [ ] Scalar power (`**`) implemented with correct forward and backward
- [ ] Scalar interoperability works for all operations
- [ ] `backward()` initializes root gradient to 1.0
- [ ] Reverse topological traversal propagates gradients correctly
- [ ] Gradient accumulation works for shared/branching nodes
- [ ] Numerical gradient checking passes for representative expressions
- [ ] All unit tests pass
- [ ] No forbidden autodiff frameworks used
- [ ] Documentation accurately reflects implementation

### Expected Artifacts

- `src/neuralearn/value.py` — Value class with forward ops and backward
- `tests/test_value.py` — Comprehensive tests
- Updated documentation

---

## Stage 2 — Extended Autodiff Operations

**Status:** IN PROGRESS  
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

- [ ] Division (`__truediv__`, `__rtruediv__`) implemented with correct forward and backward
- [ ] Exp (`exp()`) implemented with correct forward and backward
- [ ] Log (`log()`) implemented with correct forward and backward (domain: x > 0)
- [ ] Tanh (`tanh()`) implemented with correct forward and backward
- [ ] ReLU (`relu()`) implemented with correct forward and backward
- [ ] All operations have backward functions that follow the chain rule
- [ ] Numerical gradient checking passes for all new operations
- [ ] Scalar interoperability works for all new operations
- [ ] Domain/edge-case behavior documented (log domain, ReLU at zero, division by zero)
- [ ] ReLU at zero convention explicitly chosen and documented
- [ ] Chained/branching/shared graphs work correctly with new operations
- [ ] Repeated backward (D19) works correctly with new operations
- [ ] All existing Stage 1 tests still pass (zero regressions)
- [ ] New tests are added to `tests/test_value.py`
- [ ] No forbidden autodiff frameworks used
- [ ] Documentation accurately reflects implementation

### Expected Artifacts

- `src/neuralearn/value.py` — Value class extended with new operations
- `tests/test_value.py` — Comprehensive Stage 2 tests added
- Updated documentation

---

## Stage 3 — Gradient Checking

**Status:** PLANNED  
**Dependencies:** Stage 2

### Objective

Verify autodiff correctness through numerical gradient checking.

### Major Work

- Implement finite-difference gradient checking
- Create gradient-checking utilities
- Verify all implemented gradients
- Document gradient-checking methodology

### Expected Learning

- How numerical gradient approximation works
- Why analytical and numerical gradients may differ
- Appropriate tolerances for gradient checking

### Acceptance Criteria

- [ ] Finite-difference gradient checking is implemented
- [ ] All implemented gradients pass gradient checking
- [ ] Appropriate tolerance is documented
- [ ] Gradient-checking methodology is documented

### Expected Artifacts

- `src/neuralearn/gradient_check.py`
- `tests/test_gradient_check.py`
- Updated documentation

---

## Stage 4 — Parameters and Layers

**Status:** PLANNED  
**Dependencies:** Stage 2

### Objective

Implement trainable parameters and neural-network layers.

### Major Work

- Implement Parameter class
- Implement Linear layer
- Implement ReLU activation
- Implement Tanh activation
- Implement Sequential container

### Expected Learning

- How parameters differ from values
- How layers compose operations
- How forward pass builds computational graph

### Acceptance Criteria

- [ ] Parameter class stores and tracks gradients
- [ ] Linear layer computes forward pass correctly
- [ ] ReLU and Tanh layers compute correctly
- [ ] Sequential container manages layers
- [ ] Unit tests pass for all layers

### Expected Artifacts

- `src/neuralearn/parameter.py`
- `src/neuralearn/layers.py`
- `tests/test_parameter.py`
- `tests/test_layers.py`

---

## Stage 5 — Loss Functions

**Status:** PLANNED  
**Dependencies:** Stage 2

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

## Stage 7 — Training Infrastructure

**Status:** PLANNED  
**Dependencies:** Stages 4, 5, 6

### Objective

Implement the training loop and supporting infrastructure.

### Major Work

- Implement training loop
- Implement metrics computation
- Implement logging
- Implement checkpointing where appropriate

### Expected Learning

- How training loop orchestrates components
- Why gradient reset is necessary
- How metrics track training progress

### Acceptance Criteria

- [ ] Training loop completes forward/backward/update cycle
- [ ] Metrics are computed correctly
- [ ] Training produces decreasing loss
- [ ] Code is modular and testable

### Expected Artifacts

- `src/neuralearn/training.py`
- `tests/test_training.py`

---

## Stage 8 — XOR Experiment

**Status:** PLANNED  
**Dependencies:** Stage 7

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

## Stage 9 — Nonlinear Regression

**Status:** PLANNED  
**Dependencies:** Stage 7

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
