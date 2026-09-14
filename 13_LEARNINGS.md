# 13 — Learning Log

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 1 Complete — Awaiting Architect Review

## Overview

This document records learning throughout the project.

**Format:** Each entry includes Concept, Initial Understanding, What Was Implemented, What Was Observed, What Was Initially Wrong, What Was Learned, Mathematical Insight, Engineering Insight, and Open Questions.

---

## Template

Use this template for each learning entry:

```markdown
## [Concept Name]

### Initial Understanding

[What I thought I knew before implementing]

### What Was Implemented

[What I actually built]

### What Was Observed

[What happened when I ran it]

### What Was Initially Wrong

[What I got wrong initially, if anything]

### What Was Learned

[What I understand now that I didn't before]

### Mathematical Insight

[Mathematical understanding gained]

### Engineering Insight

[Engineering understanding gained]

### Open Questions

[Questions that remain]
```

---

## Learning Records

### Numerical Gradient Checking: Closures vs Fresh Instances

**Date:** 2026-09-15  
**Stage:** 1  
**Related Code:** `tests/test_value.py`

#### Initial Understanding

Numerical gradient checking requires perturbing an input and measuring the change in output. I initially wrote test functions that created new `Value` objects internally.

#### What Was Implemented

```python
# BROKEN — creates new Values each call
def fn():
    a = Value(2.0)
    b = Value(3.0)
    return (a + b).data

a = Value(2.0)
numerical_grad(fn, a)  # perturbing outer a has no effect!
```

#### What Was Observed

The numerical gradient was always 0.0, because `fn()` created independent `Value` objects. Perturbing the outer `a` did not affect the computation inside `fn()`.

#### What Was Learned

Numerical gradient checking requires the function and the perturbation target to share the same `Value` objects. Use closures:

```python
a = Value(2.0)
b = Value(3.0)
def fn():
    return (a + b).data  # captures outer a, b
numerical_grad(fn, a)    # perturbing outer a affects fn()
```

#### Mathematical Insight

Finite difference: `(f(x+eps) - f(x-eps)) / (2*eps)` approximates `df/dx`. The function `f` must depend on the same variable being perturbed.

#### Engineering Insight

Test utilities that take functions as arguments require careful design of what the function captures. Creating fresh instances inside the function breaks the connection to the perturbation target.

---

### Variable Rebinding and Computational Graphs

**Date:** 2026-09-15  
**Stage:** 1  
**Related Code:** `tests/test_value.py::TestEdgeCases::test_deep_mul_chain`

#### Initial Understanding

I assumed `a = a * 2.0` in a loop would lose the chain connection.

#### What Was Observed

The chain actually works correctly. Python keeps old `Value` objects alive because `_prev` references them. The graph is correctly connected.

#### What Was Learned

Variable rebinding (`a = a * 2.0`) does not break graph connectivity. The old node remains referenced by `_prev` of the new node. However, you must use named variables to access intermediate nodes for testing their gradients.

#### Mathematical Insight

Each `a = a * 2.0` creates a new node in the graph. The graph is a DAG where each node points to its parents. The original leaf node is always reachable through `_prev` chains.

---

### Gradient Accumulation Through Shared Nodes

**Date:** 2026-09-15  
**Stage:** 1  
**Related Code:** `tests/test_value.py::TestBranching`

#### What Was Implemented

Tests for branching graphs: `x*x + x`, `x*x*x`, `(x+x)*(x+x)`.

#### What Was Observed

When a node contributes to the output through multiple paths, its gradient is the sum of contributions from each path. The `+=` in backward rules ensures accumulation, not overwriting.

#### What Was Learned

Gradient accumulation is critical for correct autodiff. If a node is used twice, its gradient must be the sum of the gradients from both uses. The `backward()` function's use of `grad +=` (not `grad =`) in each operation's backward rule handles this automatically.

#### Mathematical Insight

For `z = f(g(x), h(x))`:
```
dz/dx = (∂f/∂g)(dg/dx) + (∂f/∂h)(dh/dx)
```
Each path contributes independently; the total gradient is the sum.

---

### Repeated Backward Calls and Stale Intermediate Gradients

**Date:** 2026-09-15  
**Stage:** 1  
**Related Code:** `src/neuralearn/value.py:202`

#### Initial Understanding

I assumed that resetting only `self.grad = 1.0` on the root before traversal was sufficient for repeated backward calls. The existing leaf-only test (`b = a * 3`) passed, confirming accumulation for simple graphs.

#### What Was Observed

For a graph with an intermediate node — `z = y * y` where `y = x * x` — the second backward call produced `x.grad = 96.0` instead of the correct `64.0`. The stale `y.grad = 8.0` from the first call was reused, causing the second traversal to add incorrect contributions.

#### What Was Learned

**Repeated backward requires resetting intermediate (non-leaf) node gradients to 0 before each traversal.** Leaf node gradients must be left untouched so they accumulate across calls. The root gradient is always set to 1.0 per call.

The three categories of nodes:
1. **Root:** Set to 1.0 each call (d(output)/d(output) = 1)
2. **Intermediate (non-leaf, non-root):** Reset to 0 each call (fresh computation)
3. **Leaf (no parents):** Untouched (accumulate across calls)

#### Mathematical Insight

For `z = f(x)` where `x` is a leaf and there are intermediate nodes `y_i`:

- First backward: `x.grad = ∂z/∂x` (computed via chain rule through fresh intermediates)
- Second backward: `x.grad = ∂z/∂x + ∂z/∂x = 2·∂z/∂x` (fresh intermediates, accumulated leaf)

If intermediates are NOT reset, the stale values corrupt the chain rule computation, producing wrong leaf gradients.

#### Engineering Insight

The existing test (`test_repeated_backward_calls`) was insufficient because it used a leaf-only graph. Testing repeated backward requires graphs with depth > 1 and intermediate nodes. A simple `b = a * 3` graph has no intermediates, so it passes even with the bug.

---

## Project-Level Learning Objectives

The following are high-level learning objectives for the project. Specific learning entries will be added as implementation progresses.

### Objective 1: Automatic Differentiation

**Goal:** Understand and implement reverse-mode automatic differentiation.

**Topics:**
- Computational graphs
- Forward pass and backward pass
- Chain rule application
- Topological ordering
- Gradient accumulation

**Success Criteria:**
- Can implement autodiff from scratch
- Can explain why each component is necessary
- Can debug autodiff issues

### Objective 2: Neural Network Training

**Goal:** Understand the complete training pipeline.

**Topics:**
- Forward pass
- Loss computation
- Backward pass
- Parameter updates
- Training dynamics

**Success Criteria:**
- Can implement training loop from scratch
- Can diagnose training issues
- Can explain why training works

### Objective 3: Optimization

**Goal:** Understand how different optimizers work.

**Topics:**
- Gradient descent
- Momentum
- Adaptive learning rates
- Learning rate effects

**Success Criteria:**
- Can implement optimizers from scratch
- Can compare optimizer behavior
- Can choose appropriate optimizer

### Objective 4: Engineering Practices

**Goal:** Develop strong engineering practices for ML.

**Topics:**
- Testing
- Documentation
- Reproducibility
- Modularity

**Success Criteria:**
- Can write comprehensive tests
- Can document design decisions
- Can ensure reproducibility

---

## Learning Entry Template (Detailed)

```markdown
## [Concept Name]

**Date:** [YYYY-MM-DD]  
**Stage:** [Stage Number]  
**Related Code:** [File and line numbers]

### Initial Understanding

[What I thought I knew before implementing]

### Research

[What I read or studied]

### What Was Implemented

[What I actually built]

### What Was Observed

[What happened when I ran it]

### What Was Initially Wrong

[What I got wrong initially, if anything]

### Debugging Process

[How I found and fixed issues]

### What Was Learned

[What I understand now that I didn't before]

### Mathematical Insight

[Mathematical understanding gained]

### Engineering Insight

[Engineering understanding gained]

### Connections

[How this connects to other concepts]

### Open Questions

[Questions that remain]

### References

[Sources consulted]
```

---

## Learning Categories

### Mathematical Concepts

- Derivatives
- Partial derivatives
- Chain rule
- Computational graphs
- Gradients
- Optimization theory

### Implementation Concepts

- Value/Tensor design
- Operation registration
- Graph construction
- Backward pass
- Gradient checking

### ML Concepts

- Backpropagation
- Gradient descent
- Loss functions
- Activation functions
- Overfitting
- Generalization

### Engineering Concepts

- Testing
- Documentation
- Reproducibility
- Modularity
- Code quality

---

## Progress Tracking

| Category | Entries | Last Updated |
|----------|---------|--------------|
| Mathematical | 2 | 2026-09-15 |
| Implementation | 2 | 2026-09-15 |
| ML | 0 | - |
| Engineering | 1 | 2026-09-15 |
| **Total** | **5** | 2026-09-15 |

---

## Future Entries

As implementation progresses, learning entries will be added here.

Each entry will follow the template above and include:
- Honest account of initial understanding
- What was actually implemented
- What was observed
- What was learned
- Mathematical and engineering insights
- Open questions
