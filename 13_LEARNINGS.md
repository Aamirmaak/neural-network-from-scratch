# 13 — Learning Log

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 8 IN PROGRESS

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

### Division Backward and the Quotient Rule

**Date:** 2026-09-15  
**Stage:** 2  
**Related Code:** `src/neuralearn/value.py` (__truediv__)

#### Initial Understanding

I initially thought division could be composed as `x * (y ** -1)`, leveraging the existing multiplication and power operations.

#### What Was Implemented

A direct `__truediv__` operation with its own backward rule:
- `∂(x/y)/∂x = 1/y`
- `∂(x/y)/∂y = -x/y²`

#### What Was Observed

The direct implementation produces a cleaner graph (one division node vs. a power + multiplication node) and simpler gradient computation.

#### What Was Learned

The quotient rule is the natural backward rule for division. Composing as `x * y**-1` would work but adds unnecessary graph nodes and makes debugging harder. The direct implementation is both more efficient and more educational.

#### Mathematical Insight

For `z = x/y`:
- `∂z/∂x = 1/y` (numerator gradient)
- `∂z/∂y = -x/y²` (denominator gradient, negative because larger denominator → smaller result)

#### Engineering Insight

Sometimes the "composed" approach (building from existing primitives) is less efficient and less clear than a dedicated implementation. The trade-off is code size vs. clarity and performance.

---

### Non-Smooth Functions: ReLU at Zero

**Date:** 2026-09-15  
**Stage:** 2  
**Related Code:** `src/neuralearn/value.py` (relu)

#### Initial Understanding

ReLU is `max(0, x)`, which is differentiable everywhere except at x=0.

#### What Was Implemented

ReLU with gradient=0 at x=0 (standard deep learning convention).

#### What Was Observed

The convention works correctly in all tests, including repeated backward calls.

#### What Was Learned

Non-smooth functions require an explicit convention at the non-differentiable point. The choice (gradient=0 vs gradient=1 vs gradient=0.5) affects training behavior but all are valid subgradient choices. Gradient=0 is standard because it means "don't update" at the boundary.

#### Mathematical Insight

ReLU is not differentiable at x=0 in the classical sense, but it has subgradients. The subdifferential at x=0 is [0, 1]. Choosing 0 is the most conservative choice — it means the neuron doesn't update when exactly at zero.

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

### Loss Functions as Functions vs Modules

**Date:** 2026-09-15  
**Stage:** 5  
**Related Code:** `src/neuralearn/losses.py`

#### Initial Understanding

I initially considered making loss functions Module subclasses, consistent with layers.

#### What Was Implemented

Plain functions: `mse_loss(predictions, targets)` and `binary_cross_entropy(predictions, targets)`.

#### What Was Learned

Loss functions are fundamentally different from layers:
1. They have no trainable parameters — `parameters()` would return `[]`
2. They have no state — no need for `zero_grad()`
3. They are pure computations: input → scalar output
4. Plain functions are simpler, more Pythonic, and easier to test

The key insight: Module adds value when you need parameter discovery and gradient management. Loss functions need neither.

#### Mathematical Insight

MSE gradient: `dMSE/dŷ_i = 2(ŷ_i - y_i) / n` — simple, always defined.
BCE gradient: `dBCE/dp_i = (1/n) * [-y_i/p_i + (1-y_i)/(1-p_i)]` — requires 0 < p < 1.

#### Engineering Insight

When designing APIs, ask: "Does this entity need the abstraction's features?" If not, use the simpler form. Functions for stateless computations, classes for stateful ones.

---

### BCE Numerical Stability Through Clipping

**Date:** 2026-09-15  
**Stage:** 5  
**Related Code:** `src/neuralearn/losses.py` (binary_cross_entropy)

#### Initial Understanding

I initially thought BCE required special-case handling for p=0 and p=1 (if/else branches).

#### What Was Implemented

Clipping using existing relu(): `clipped = eps + (p - eps).relu() - (p - (1-eps)).relu()`

#### What Was Learned

Composing clipping from existing differentiable operations is better than if/else:
1. No Python branching in the computational graph
2. Clipping is itself differentiable (gradient 0 at boundaries, 1 in between)
3. Uses existing Value operations — no new backward rules needed
4. The non-differentiability at exact boundaries is handled by the same convention as ReLU (gradient = 0)

#### Mathematical Insight

The clipping formula `eps + max(0, p-eps) - max(0, p-(1-eps))` implements:
- p < eps: output = eps (gradient = 0)
- eps <= p <= 1-eps: output = p (gradient = 1)
- p > 1-eps: output = 1-eps (gradient = 0)

This is the subgradient of the projection onto [eps, 1-eps].

#### Engineering Insight

Reusing existing differentiable operations (like relu) to build new differentiable operations is powerful. It avoids implementing new backward rules and ensures consistency with the existing gradient-checking infrastructure.

---

### BCE Requires Sigmoid Activation Before the Loss

**Date:** 2026-09-16  
**Stage:** 7  
**Related Code:** `tests/test_training.py::TestBCETraining`

#### Initial Understanding

I initially wrote a BCE training test using raw Linear output directly as input to binary_cross_entropy, expecting the loss to decrease.

#### What Was Observed

The BCE loss was stuck at a constant value (8.059) and did not decrease across 200 epochs. The model parameters were not updating.

#### What Was Learned

BCE expects predictions to be probabilities in (0, 1). Raw Linear output can be any real number. The BCE clipping mechanism (`eps + (p-eps).relu() - (p-(1-eps)).relu()`) clips predictions to [eps, 1-eps], but when raw outputs are far outside this range, the gradient through the clipping is 0 (dead zone). This means:

- If raw output < eps: gradient = 0 (clipped to eps)
- If raw output > 1-eps: gradient = 0 (clipped to 1-eps)
- Only in (eps, 1-eps) does the gradient flow

The solution is to apply sigmoid to the raw output before passing to BCE, converting any real number to (0, 1). This is the standard pattern: `model → sigmoid → BCE`.

#### Mathematical Insight

sigmoid(x) = 1 / (1 + exp(-x)) maps (-∞, ∞) → (0, 1). This is the natural "probability output" for binary classification. The gradient of BCE with sigmoid input simplifies to `pred - target`, which is numerically stable and always nonzero for wrong predictions.

#### Engineering Insight

Loss functions assume specific input ranges. MSE assumes any real number; BCE assumes (0, 1). The activation function before the loss must produce values in the expected range. This is a fundamental design pattern in neural networks, not an implementation detail.

---

### Optimizer State Management via Object Identity

**Date:** 2026-09-15  
**Stage:** 6  
**Related Code:** `src/neuralearn/optimizers.py`

#### Initial Understanding

I initially considered using parameter values as dictionary keys for optimizer state.

#### What Was Implemented

State keyed by `id(parameter)`, which is the object's memory address.

#### What Was Learned

Using `id()` for state management is robust because:
1. `id()` is stable for the lifetime of an object
2. Avoids hash/equality issues with Value objects (which don't implement `__hash__`)
3. Two Parameters with the same numeric value but different identity correctly get separate state
4. State is naturally lost when a Parameter is garbage collected (acceptable behavior)

The key insight: optimizer state is inherently tied to object identity, not object value. A Parameter's value changes every step, but its identity remains constant.

#### Engineering Insight

When designing stateful systems that associate state with objects, prefer object identity (`id()`) over object equality. This avoids subtle bugs where equal-valued but distinct objects share state incorrectly.

---

### Adam Bias Correction Mathematics

**Date:** 2026-09-15  
**Stage:** 6  
**Related Code:** `src/neuralearn/optimizers.py` (Adam.step)

#### Initial Understanding

I knew Adam uses bias-corrected moments but had to verify the exact equations.

#### What Was Implemented

Standard Adam with:
- `m_t = beta1 * m_{t-1} + (1-beta1) * g_t`
- `v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2`
- `m_hat = m_t / (1 - beta1^t)`
- `v_hat = v_t / (1 - beta2^t)`

#### What Was Learned

Bias correction is critical in early steps:
- At t=1 with beta1=0.9: m_hat = m / 0.1 = 10*m (amplifies the first gradient by 10x)
- At t=1 with beta2=0.999: v_hat = v / 0.001 = 1000*v (amplifies the first squared gradient by 1000x)
- This compensates for the initialization of m=0, v=0 which would otherwise produce very small updates

#### Mathematical Insight

The bias correction terms `(1 - beta^t)` approach 1 as t grows, so the correction vanishes for large t. This means Adam behaves like a well-calibrated adaptive learning rate method after the initial warm-up phase.

---

### Dataset/DataLoader Separation of Concerns

**Date:** 2026-09-16  
**Stage:** 8  
**Related Code:** `src/neuralearn/datasets.py`, `src/neuralearn/dataloaders.py`

#### Initial Understanding

I initially considered a single class that handled both data storage and iteration.

#### What Was Implemented

Two separate classes: Dataset (storage + validation) and DataLoader (iteration + batching + shuffling).

#### What Was Learned

Separating storage from iteration is cleaner because:
1. Dataset is a simple, stateless container — easy to test and reason about
2. DataLoader handles the complex iteration logic (batching, shuffling, seeds)
3. Different DataLoader configurations can iterate over the same Dataset
4. The separation mirrors how real frameworks (PyTorch) organize data loading
5. Each class has a single, clear responsibility

The key insight: Dataset is to data what a list is to elements — a container with indexing. DataLoader is to Dataset what a range-based for loop is to a list — an iteration mechanism.

#### Mathematical Insight

Batch count: `ceil(n/batch_size)` without drop_last, `floor(n/batch_size)` with drop_last. This is a simple ceiling/floor function but getting it right for edge cases (partial batches, dataset smaller than batch) requires careful thought.

#### Engineering Insight

Deterministic shuffling requires a dedicated random.Random instance per DataLoader, not the global random module. This avoids polluting global state and ensures two DataLoaders with the same seed produce identical behavior.

---

### Per-Sample Training Within Batches

**Date:** 2026-09-16  
**Stage:** 8  
**Related Code:** `src/neuralearn/training.py`

#### Initial Understanding

I initially thought DataLoader should accumulate gradients across a batch and call optimizer.step() once per batch.

#### What Was Implemented

Per-sample training within batches: each sample triggers forward → loss → backward → step → zero_grad, even when batched.

#### What Was Learned

For the scalar architecture, per-sample training is the only correct approach because:
1. Loss functions operate on individual predictions, not batch tensors
2. There is no batched matrix multiplication
3. Gradient accumulation would require careful loss scaling
4. Per-sample training is mathematically equivalent to batch training with batch_size=1

The DataLoader batches are for iteration grouping and shuffling, not for gradient computation. This preserves D39 (per-sample training) while adding data loading infrastructure.

#### Engineering Insight

When designing DataLoader integration with a training engine, the critical question is: "Does the DataLoader change the gradient/update semantics?" If yes, it's a major architectural change. If no (as in our case), it's a clean extension that preserves existing behavior.

---

### Visualization as Optional Analysis Layer

**Date:** 2026-09-16  
**Stage:** 10  
**Related Code:** `src/neuralearn/visualization.py`

#### Initial Understanding

I initially considered integrating plotting into the Trainer class.

#### What Was Implemented

A separate `visualization.py` module with standalone plotting functions. matplotlib is imported only when visualization functions are called.

#### What Was Learned

Keeping visualization separate is cleaner because:
1. Core framework has zero visualization dependencies
2. Plotting functions are pure utilities — they read data and create files
3. Users who don't need matplotlib aren't affected
4. The Trainer class stays focused on training, not presentation
5. Different visualization approaches can be tried without touching core code

The key insight: visualization is a consumer of experiment data, not a producer of it. It belongs in the analysis layer, not the training layer.

#### Engineering Insight

For optional dependencies, use lazy imports (import inside the function). This way the core package works without the dependency, and users get a clear error message if they try to use visualization without matplotlib installed.

---

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
- Dataset/DataLoader design

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

### Central-Difference Finite Differences for Gradient Checking

**Date:** 2026-09-15  
**Stage:** 3  
**Related Code:** `src/neuralearn/gradient_check.py`

#### Initial Understanding

I initially thought any finite-difference method would work for gradient checking. I considered forward difference: `(f(x+ε) - f(x)) / ε`.

#### What Was Implemented

Central difference: `[f(x+ε) - f(x-ε)] / (2ε)` with ε=1e-5.

#### What Was Observed

Central difference produces accurate numerical gradients that closely match analytical gradients for all smooth operations. The error is typically < 1e-7.

#### What Was Learned

Central difference has O(ε²) truncation error, compared to O(ε) for forward difference. This means it's significantly more accurate for the same ε. The trade-off is two function evaluations instead of one, which is acceptable for gradient checking (not a hot path).

For ε=1e-5:
- Truncation error: O(ε²) ≈ 1e-10
- Roundoff error: O(machine_epsilon/ε) ≈ 1e-11
- Total error: dominated by whichever is larger, both are small

#### Mathematical Insight

Forward difference: `f'(x) ≈ (f(x+h) - f(x)) / h` has error `O(h)` from the Taylor expansion remainder.
Central difference: `f'(x) ≈ (f(x+h) - f(x-h)) / (2h)` has error `O(h²)` because the first-order terms cancel.

#### Engineering Insight

The ε selection is a balance: too large loses accuracy, too small amplifies floating-point roundoff. ε=1e-5 is standard in the autodiff literature and works well for float64.

---

### Gradient Checking Independence from Autodiff

**Date:** 2026-09-15  
**Stage:** 3  
**Related Code:** `src/neuralearn/gradient_check.py`

#### Initial Understanding

I initially considered having the gradient checker call `backward()` internally to compute analytical gradients automatically.

#### What Was Implemented

The `gradient_check` function reads existing `.grad` values (computed by the caller) and computes numerical gradients independently. It does NOT call `backward()`.

#### What Was Observed

This design keeps the numerical and analytical computations conceptually independent. The caller controls when `backward()` is called.

#### What Was Learned

If the gradient checker called `backward()` internally, it would:
1. Couple the checking logic to the Value API
2. Risk modifying gradients that the caller wants to preserve
3. Make it impossible to check gradients after custom gradient manipulation

Reading existing `.grad` values is cleaner and more composable.

#### Engineering Insight

Separation of concerns: the gradient checker validates gradients; the autodiff engine computes them. The checker should not perform the computation it's trying to validate.

---

### Parameter Subclassing Value for Clean Graph Participation

**Date:** 2026-09-15  
**Stage:** 4  
**Related Code:** `src/neuralearn/parameter.py`

#### What Was Implemented

Parameter subclasses Value directly. It adds `requires_grad` and `zero_grad()` but inherits all arithmetic operations.

#### What Was Learned

Subclassing Value (IS-A) was clearly better than wrapping Value (HAS-A):
1. No adapter code needed — `Parameter * Value` just works
2. The computational graph naturally includes Parameters as leaf nodes
3. `backward()` correctly accumulates gradients into Parameters
4. Optimizers can simply iterate `.parameters()` and read/write `.data`/`.grad`

The key insight: since Value already has the complete interface (data, grad, arithmetic ops), subclassing adds zero ceremony. A wrapper would have needed `__mul__`, `__add__`, etc. delegation methods.

#### Engineering Insight

The `__call__` pattern on Module makes layer usage clean: `y = layer(x)` instead of `y = layer.forward(x)`. This matches the PyTorch convention and feels natural.

---

### Module as Convention, Not Abstraction

**Date:** 2026-09-15  
**Stage:** 4  
**Related Code:** `src/neuralearn/layers.py`

#### What Was Implemented

Module is a base class with `forward()`, `parameters()`, `zero_grad()`. It is NOT abstract — no metaclass, no ABC, no decorator magic.

#### What Was Learned

For an educational framework, a plain base class is better than an abstract base class:
1. Students can read and understand it immediately
2. `NotImplementedError` in `forward()` gives a clear error message
3. `parameters()` returns `[]` by default, which is correct for stateless modules
4. `zero_grad()` uses `parameters()` — no duplication

The simplicity means: when someone reads `class Linear(Module)`, they immediately understand the interface without needing to learn framework conventions.

---

## Progress Tracking

| Category | Entries | Last Updated |
|----------|---------|--------------|
| Mathematical | 6 | 2026-09-15 |
| Implementation | 5 | 2026-09-15 |
| ML | 3 | 2026-09-16 |
| Engineering | 10 | 2026-09-16 |
| **Total** | **24** | 2026-09-16 |

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
