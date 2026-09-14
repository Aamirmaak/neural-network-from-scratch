# 04 — Architecture

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 2 In Progress

## Overview

This document describes the architecture of the neural-network framework. The architecture is designed for educational clarity, correctness, and modularity.

**Current State:** Stage 2 in progress — scalar `Value` class extended with division, exp, log, tanh, ReLU. Layers, losses, optimizers, and training are planned but not yet implemented.

## Conceptual Architecture

The system follows a layered architecture where each layer depends only on layers below it:

```
Experiments
    ↑
Visualization
    ↑
Training Loop
    ↑
Optimizer
    ↑
Loss
    ↑
Model
    ↑
Layers
    ↑
Parameters
    ↑
Autodiff
    ↑
Computational Graph
    ↑
Value / Tensor
```

## Module Responsibilities

### Value (Stage 1 — Implemented)

**Purpose:** Store a scalar numerical value with gradient tracking and computational graph metadata.

**Responsibilities:**
- Hold a single scalar numerical value (`data`)
- Store accumulated gradient with respect to the output (`grad`)
- Track parent nodes in the computational graph (`_prev`)
- Record the operation that produced this value (`_op`)
- Store a local backward function that computes parent gradients (`_backward`)

**Key Properties:**

```python
class Value:
    data: float        # The scalar numerical value
    grad: float        # Accumulated gradient (initialized to 0.0)
    _prev: set         # Parent Value nodes
    _op: str           # Operation name (e.g., '+', '*', '**')
    _backward: fn      # Local backward rule (computes parent gradients)
```

**Stage 1 Supported Operations:**

| Operation | Forward | Local Derivative | Backward Rule |
|-----------|---------|------------------|---------------|
| Addition | `z = x + y` | `∂z/∂x = 1`, `∂z/∂y = 1` | `dx += dout`, `dy += dout` |
| Multiplication | `z = x * y` | `∂z/∂x = y`, `∂z/∂y = x` | `dx += y * dout`, `dy += x * dout` |
| Negation | `z = -x` | `∂z/∂x = -1` | `dx += -1 * dout` |
| Subtraction | `z = x - y` | `∂z/∂x = 1`, `∂z/∂y = -1` | `dx += dout`, `dy += -dout` |
| Power | `z = x ** n` | `∂z/∂x = n * x^(n-1)` | `dx += n * x^(n-1) * dout` |

**Stage 2 Additional Operations:**

| Operation | Forward | Local Derivative | Backward Rule |
|-----------|---------|------------------|---------------|
| Division | `z = x / y` | `∂z/∂x = 1/y`, `∂z/∂y = -x/y²` | `dx += dout / y`, `dy += dout * (-x/y²)` |
| Reciprocal | `z = 1 / x` | `∂z/∂x = -1/x²` | `dx += dout * (-1/x²)` |
| Exp | `z = exp(x)` | `∂z/∂x = exp(x)` | `dx += dout * exp(x)` |
| Log | `z = log(x)` | `∂z/∂x = 1/x` | `dx += dout * (1/x)` |
| Tanh | `z = tanh(x)` | `∂z/∂x = 1 - tanh(x)²` | `dx += dout * (1 - tanh(x)²)` |
| ReLU | `z = max(0, x)` | `∂z/∂x = 1 if x > 0, 0 if x < 0, convention at x=0` | `dx += dout * (1 if x > 0 else 0)` |

**Scalar Interoperability:** Operations between `Value` objects and Python numeric scalars are supported (e.g., `x + 2`, `3 * x`, `x ** 2`).

**Power Domain (Stage 1):**

| Case | Supported | Notes |
|------|-----------|-------|
| `x > 0`, any `n` | Yes | Standard domain |
| `x = 0`, `n > 0` | Yes | `0^n = 0`, derivative `n * 0^(n-1)` is 0 when `n > 1` |
| `x < 0`, integer `n` | Yes | Negative base with integer exponent is well-defined |
| `x < 0`, non-integer `n` | No | Produces complex values; caller's responsibility |
| `x = 0`, `n = 0` | Edge case | Python defines `0**0 = 1`; derivative is 0 |
| `x = 0`, `n < 0` | No | Division by zero; produces `inf`/`nan` |

No explicit domain checks are enforced; the caller is responsible for providing valid inputs.

### Computational Graph (Stage 1 — Implemented)

**Purpose:** Record the sequence of operations that produced a value, forming a directed acyclic graph (DAG).

**Responsibilities:**
- Build graph as operations are performed (each operation creates a new `Value` node)
- Track dependencies via `_prev` (parent references on each node)
- Enable topological sorting for backward pass

**Key Concept:** Every arithmetic operation on `Value` objects creates a new `Value` node whose `_prev` contains the input nodes. This naturally builds a DAG.

**Graph Construction Example:**

```text
a = Value(2.0)          # leaf node, _prev = {}
b = Value(3.0)          # leaf node, _prev = {}
c = a * b               # c._prev = {a, b}, c._op = '*'
d = a + b               # d._prev = {a, b}, d._op = '+'
e = c + d               # e._prev = {c, d}, e._op = '+'
```

### Reverse-Mode Automatic Differentiation (Stage 1 — Implemented)

**Purpose:** Compute exact gradients through the computational graph by traversing it in reverse.

**Responsibilities:**
- Initialize the output node's gradient to 1.0 (`d(output)/d(output) = 1`)
- Build reverse topological order from the output node
- Traverse nodes in reverse topological order
- Execute each node's local backward rule to compute parent gradients
- Accumulate gradients into parent nodes (add, don't overwrite)

**Algorithm:**

```text
output.backward():
    1. Build topological order by DFS from output
    2. Reset gradients:
         - root  → 1.0
         - intermediate (non-root, non-leaf) → 0.0
         - leaf  → untouched (accumulate across calls)
    3. For each node in REVERSE topological order:
         node._backward(node.grad)
             → for each parent, add parent's gradient contribution
```

**Why Topological Ordering:** Ensures that when we process a node, the gradient flowing into it from all downstream paths has already been fully accumulated. Without this ordering, we might process a node before its gradient is finalized.

**Gradient Accumulation:** When a node contributes to the output through multiple paths, its gradient is the sum of contributions from each path. This is handled by initializing `grad = 0` and using `+=` in backward rules.

**Repeated Backward Calls:** Each `backward()` call computes fresh gradients and adds them to leaf-node gradients. Intermediate (non-leaf) gradients are reset to 0 before each traversal to prevent stale values from corrupting the current pass. This means:
- Leaf gradients accumulate across calls (useful for gradient accumulation)
- Intermediate gradients are freshly computed each time
- Root gradient is always 1.0 per call

```text
z = x*x + x

Graph:
  x → x*x → z
  x ↗

Backward:
  dz/d(x*x) = 1,  dz/dx (from +) = 1
  d(x*x)/dx = 2x (since x*x → 2x * dout)
  Total dx = 2x + 1
```

### Parameters

**Purpose:** Store trainable model weights.

**Responsibilities:**
- Store weight data
- Track gradients
- Be distinguishable from non-trainable values

**Key Difference from Value:** Parameters are the values we want to optimize. They are the "leaf nodes" of the computational graph.

### Layers

**Purpose:** Compose operations into reusable building blocks.

**Responsibilities:**
- Define forward computation
- Manage parameters
- Connect to autodiff system

**Key Layers:**
- Linear: `y = Wx + b`
- ReLU: `y = max(0, x)`
- Tanh: `y = tanh(x)`

### Model

**Purpose:** Container for layers that defines a complete computation.

**Responsibilities:**
- Stack layers
- Define forward pass
- Collect parameters

### Loss

**Purpose:** Measure how far predictions are from targets.

**Responsibilities:**
- Compute scalar loss from predictions and targets
- Enable gradient computation

### Optimizer

**Purpose:** Update parameters based on gradients.

**Responsibilities:**
- Compute parameter updates
- Apply updates to parameters
- Manage state (momentum, adaptive rates)

### Training Loop

**Purpose:** Orchestrate the training process.

**Responsibilities:**
- Forward pass
- Loss computation
- Backward pass
- Parameter update
- Gradient reset
- Metrics computation
- Logging

### Experiments

**Purpose:** Demonstrate and investigate training behavior.

**Responsibilities:**
- Define datasets
- Configure models
- Run training
- Collect results
- Generate visualizations

## Design Decisions

### D1: Why a Value/Tensor Abstraction?

**Decision:** Implement a Value class that wraps numerical data with gradient tracking.

**Reasoning:**
- Modern autodiff systems require tracking how values are produced
- The Value class is the foundation for the computational graph
- Storing metadata (parents, operation, backward_fn) enables the backward pass

**Alternatives Considered:**
- Pure functional approach: More complex, harder to understand
- Metaclass-based: Over-engineered for educational purpose

### D2: Why Store Computational Graph Information?

**Decision:** Each Value stores its parents and the operation that created it.

**Reasoning:**
- Backpropagation requires traversing the graph backward
- Parent relationships define the graph structure
- Operation information selects the correct backward function
- This approach makes the graph explicit and inspectable

**Alternatives Considered:**
- Implicit graph via function calls: Harder to inspect, debug
- Global graph registry: Unnecessary complexity

### D3: Why Reverse-Mode Autodiff?

**Decision:** Implement reverse-mode (backpropagation) rather than forward-mode.

**Reasoning:**
- Reverse-mode is efficient for functions with many outputs (scalar loss)
- This is the standard approach in deep learning
- Educational value: this is how real frameworks work

**Trade-offs:**
- Forward-mode is simpler but inefficient for neural-network training
- Reverse-mode requires storing intermediate values (memory vs computation)

### D4: Why Topological Ordering?

**Decision:** Use topological sort for backward pass ordering.

**Reasoning:**
- Ensures gradients are computed in correct dependency order
- Required for correct chain-rule application
- Prevents using gradients before they are computed

**Alternatives Considered:**
- Recursive DFS: Risk of stack overflow for deep graphs
- Breadth-first: May not respect dependencies

### D5: Why Separate Layers from Autodiff Core?

**Decision:** Keep layers as a separate module from the autodiff core.

**Reasoning:**
- Clear separation of concerns
- Autodiff core is generic; layers are specific
- Easier to test each component independently
- Layers can be composed in different ways

**Alternatives Considered:**
- Monolithic design: Harder to understand, test, maintain

### D6: Why Optimizers Independent from Models?

**Decision:** Optimizers are separate objects that take parameters as input.

**Reasoning:**
- Different optimizers can be used with same model
- Optimizers don't need to know about model architecture
- Clear interface: `optimizer.step()` updates all parameters
- Matches how real frameworks work

**Alternatives Considered:**
- Model-owned optimizer: Couples model and optimizer

### D7: Why Experiments Separate from Library Code?

**Decision:** Experiments live in a separate directory, not in the library.

**Reasoning:**
- Library code is reusable; experiments are specific
- Experiments may have different dependencies (matplotlib, etc.)
- Keeps library code clean
- Enables multiple experiments without cluttering core

**Alternatives Considered:**
- Example scripts in library: Less organized, harder to manage

## Module Structure

```
neuralearn/
├── __init__.py          # Package initialization
├── value.py             # Value/Tensor abstraction
├── operations.py        # Arithmetic operations and backward functions
├── autodiff.py          # Topological sort and backward pass
├── gradient_check.py    # Numerical gradient checking
├── parameter.py         # Trainable parameters
├── layers.py            # Linear, ReLU, Tanh, Sequential
├── losses.py            # MSE, Binary Cross-Entropy
├── optimizers.py        # SGD, Momentum, Adam
├── training.py          # Training loop
└── visualization.py     # Plotting and visualization
```

## Data Flow

### Forward Pass

```
Input Data
    ↓
Layer 1.forward(input)
    ↓
Layer 2.forward(output1)
    ↓
...
    ↓
Layer N.forward(outputN-1)
    ↓
Predictions
    ↓
Loss(predictions, targets)
    ↓
Loss Value
```

### Backward Pass

```
Loss Value (grad = 1.0)
    ↓
Topological Sort
    ↓
For each node in reverse topological order:
    node.backward(grad_output)
        ↓
    Accumulate gradients to parents
        ↓
All parameters have gradients
```

### Optimizer Step

```
For each parameter:
    Update parameter.data based on parameter.grad
```

## Numerical Considerations

### Gradient Accumulation

When a value is used multiple times, its gradient is the sum of gradients from each use. This is handled by initializing gradients to zero and accumulating.

### Gradient Clipping

Not implemented in initial scope, but the architecture supports adding gradient clipping in the training loop.

### Numerical Stability

Operations like log and exp may require careful implementation to avoid overflow/underflow. This will be addressed during implementation.

## Testing Architecture

### Unit Tests

Each module has corresponding tests:
- `tests/test_value.py`
- `tests/test_operations.py`
- `tests/test_autodiff.py`
- `tests/test_gradient_check.py`
- `tests/test_parameter.py`
- `tests/test_layers.py`
- `tests/test_losses.py`
- `tests/test_optimizers.py`
- `tests/test_training.py`

### Integration Tests

- `tests/test_integration.py`: End-to-end training flows

### Gradient Checking

All gradients verified via finite-difference approximation.

## Visualization Architecture

### Training Curves

- Loss vs epoch
- Metrics vs epoch

### Predictions

- Regression: predicted vs actual
- Classification: decision boundary

### Experiments

- Comparison plots across configurations

## Constraints

1. **No external autodiff:** Core implementation uses no autodiff libraries
2. **NumPy only:** Numerical operations via NumPy
3. **Educational clarity:** Code prioritizes understanding over performance
4. **Modularity:** Clear interfaces between components
5. **Testability:** All components independently testable

## Open Questions

None at this stage. Architecture decisions are documented in [10_DECISIONS.md](10_DECISIONS.md).
