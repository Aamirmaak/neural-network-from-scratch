# 07 — Testing Strategy

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 5 IN PROGRESS

## Overview

This document defines the testing strategy for ensuring correctness of the neural-network framework.

**Philosophy:** A demo running successfully is NOT sufficient evidence of correctness.

## Testing Principles

1. **Correctness is critical:** Every component must be verified
2. **Multiple validation methods:** Unit tests, gradient checking, integration tests
3. **Incremental verification:** Test each component before building on it
4. **Honest reporting:** Document what was tested and what passed
5. **Reproducible tests:** Tests produce same results given same inputs

## Testing Levels

### Level 1: Unit Testing

**Purpose:** Verify individual components work correctly.

**Scope:**
- Arithmetic operations
- Graph construction
- Layer forward passes
- Loss computations
- Optimizer updates

**Method:**
- pytest framework
- Assert expected outputs for known inputs
- Test edge cases (zeros, negatives, large values)

**When:** After implementing each component

### Level 2: Gradient Checking

**Purpose:** Verify that analytical gradients match numerical approximations.

**Scope:**
- All operations with gradients
- All layers with parameters
- All loss functions

**Method:**
- Finite-difference approximation
- Compare with autodiff gradients
- Check within tolerance

**When:** After implementing each gradient computation

### Level 3: Integration Testing

**Purpose:** Verify that components work together correctly.

**Scope:**
- Forward + backward pass
- Optimizer updates
- Training loop

**Method:**
- End-to-end training on simple problems
- Verify loss decreases
- Verify parameters update correctly

**When:** After implementing training infrastructure

### Level 4: Regression Testing

**Purpose:** Ensure later changes don't break previously working code.

**Scope:**
- All previously passing tests

**Method:**
- Run full test suite after changes
- Verify no regressions

**When:** After any code change

### Level 5: Experiment Validation

**Purpose:** Verify that experiments produce reasonable results.

**Scope:**
- XOR classification
- Nonlinear regression
- Controlled experiments

**Method:**
- Run experiments with documented configurations
- Verify expected behavior
- Document results

**When:** After implementing experiments

---

## Unit Testing

### Test Organization

```
tests/
├── test_value.py           # Value class tests (Stages 1 & 2)
├── test_gradient_check.py  # Gradient-checking tests (Stage 3)
├── test_operations.py      # Arithmetic operation tests (PLANNED)
├── test_autodiff.py        # Autodiff tests (PLANNED)
├── test_parameter.py       # Parameter tests (Stage 4 — COMPLETE)
├── test_layers.py          # Layer tests (Stage 4 — COMPLETE)
├── test_losses.py          # Loss function tests (Stage 5 — IN PROGRESS)
├── test_optimizers.py      # Optimizer tests (PLANNED)
├── test_training.py        # Training loop tests (PLANNED)
└── test_integration.py     # Integration tests (PLANNED)
```

### Test Cases

#### Value Class (Stage 1)

| Test | Description |
|------|-------------|
| Creation | Value stores data correctly |
| Gradient initialization | Gradient starts at zero |
| Parent tracking | Parents are recorded correctly |
| Operation tracking | Operation string is recorded |
| String repr | Debugging representation works |

#### Arithmetic Operations (Stage 1)

| Test | Description |
|------|-------------|
| Addition | a + b produces correct result |
| Multiplication | a * b produces correct result |
| Negation | -a produces correct result |
| Subtraction | a - b produces correct result |
| Power | a ** n produces correct result |
| Scalar addition | a + scalar, scalar + a |
| Scalar multiplication | a * scalar, scalar * a |
| Scalar subtraction | a - scalar, scalar - a |
| Scalar power | a ** scalar |

#### Backward Propagation (Stage 1)

| Test | Description |
|------|-------------|
| Addition backward | Gradients flow correctly through addition |
| Multiplication backward | Gradients flow correctly through multiplication |
| Negation backward | Gradient negated correctly |
| Subtraction backward | Gradients flow correctly through subtraction |
| Power backward | Gradient n*x^(n-1) computed correctly |
| Chained expressions | Multi-step expressions produce correct gradients |
| Branching graphs | Shared nodes accumulate gradients from multiple paths |
| Root gradient | output.backward() sets output.grad = 1.0 |
| Multiple variables | Independent gradients for multiple leaf nodes |
| Numerical gradient check | Finite-difference verification of analytical gradients |
| Repeated backward (leaf-only) | Accumulation works for simple graphs |
| Repeated backward (intermediate) | Intermediate gradients reset; leaf gradients accumulate correctly |
| Repeated backward (branching) | Shared-node graphs produce correct accumulation across calls |

#### Layers

| Test | Description |
|------|-------------|
| Linear forward | y = Wx + b computes correctly |
| Linear parameters | Parameters have correct shapes |
| ReLU forward | max(0, x) computes correctly |
| Tanh forward | tanh(x) computes correctly |
| Sequential forward | Layers are called in order |

#### Losses

| Test | Description |
|------|-------------|
| MSE forward | MSE computes correctly |
| Binary CE forward | BCE computes correctly |

#### Optimizers

| Test | Description |
|------|-------------|
| SGD update | Parameters update correctly |
| Momentum update | Velocity tracks correctly |
| Adam update | Moments track correctly |

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_value.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=neuralearn
```

---

## Numerical Gradient Checking

### Purpose

Numerical gradient checking verifies that analytical gradients (computed by autodiff) match numerical approximations. This is critical for validating backward function implementations.

### Method

#### Finite-Difference Approximation

For a function f(x) and parameter θ:

```
∂f/∂θ ≈ (f(θ + ε) - f(θ - ε)) / (2ε)
```

where ε is a small perturbation (e.g., 1e-5).

#### Central Difference

Central difference is more accurate than forward difference:

```
Forward: (f(θ + ε) - f(θ)) / ε
Central: (f(θ + ε) - f(θ - ε)) / (2ε)
```

### Implementation (Stage 3 — Implemented)

```python
# src/neuralearn/gradient_check.py
def numerical_grad(fn, x_val, eps=1e-5):
    """Compute numerical gradient via central difference."""
    original = x_val.data
    x_val.data = original + eps
    forward = fn()
    x_val.data = original - eps
    backward = fn()
    x_val.data = original
    return (forward - backward) / (2.0 * eps)

def gradient_check(fn, values, eps=1e-5, atol=1e-5, rtol=1e-3):
    """Check analytical vs numerical gradients for multiple Values."""
    results = []
    for val in values:
        analytical = val.grad
        original = val.data
        val.data = original + eps
        forward = fn()
        val.data = original - eps
        backward = fn()
        val.data = original
        numerical = (forward - backward) / (2.0 * eps)
        abs_error = abs(analytical - numerical)
        denom = max(abs(analytical), abs(numerical), 1e-12)
        rel_error = abs_error / denom
        passed = (abs_error < atol) or (rel_error < rtol)
        results.append({
            'value': val, 'analytical': analytical, 'numerical': numerical,
            'abs_error': abs_error, 'rel_error': rel_error, 'passed': passed,
        })
    return results
```

### Tolerance

| Condition | Tolerance |
|-----------|-----------|
| Excellent | < 1e-7 |
| Good | < 1e-5 |
| Acceptable | < 1e-3 |
| Poor | > 1e-3 (investigate) |

### Error Sources

| Source | Description |
|--------|-------------|
| Truncation error | Approximation error from finite ε |
| Roundoff error | Floating-point precision limitations |
| Implementation error | Bug in backward function |

### When to Check

- After implementing each backward function
- After modifying any gradient computation
- When debugging training issues
- Before claiming correctness

---

## Integration Testing

### Training Flow Test

**Test:** Complete training loop on simple problem

**Steps:**
1. Create simple dataset (e.g., y = x)
2. Create small model (e.g., Linear → ReLU → Linear)
3. Train for N epochs
4. Verify loss decreases
5. Verify predictions improve

**Expected:** Loss decreases, predictions approach targets

### Optimizer Update Test

**Test:** Verify optimizer updates parameters correctly

**Steps:**
1. Create parameter with known value
2. Set known gradient
3. Run optimizer step
4. Verify parameter updated correctly

**Expected:** Parameter matches expected update

### Gradient Flow Test

**Test:** Verify gradients flow through network

**Steps:**
1. Create multi-layer network
2. Forward pass
3. Backward pass
4. Verify all parameters have gradients
5. Verify gradients are non-zero (for active parameters)

**Expected:** All parameters have non-zero gradients

---

## Regression Testing

### Purpose

Ensure later changes don't break previously working code.

### Method

1. Maintain comprehensive test suite
2. Run full test suite after any code change
3. Track test results over time
4. Investigate any test failures

### Test Suite

```bash
# Run full test suite
pytest

# Run with coverage report
pytest --cov=neuralearn --cov-report=html
```

---

## Experiment Validation

### XOR Experiment Validation

**Checklist:**
- [ ] Model trains without errors
- [ ] Loss decreases over epochs
- [ ] Final accuracy > 90%
- [ ] Decision boundary is nonlinear
- [ ] Results are reproducible with same seed

### Regression Experiment Validation

**Checklist:**
- [ ] Model trains without errors
- [ ] Loss decreases over epochs
- [ ] Predictions follow target function
- [ ] Results are reproducible with same seed

---

## Test Data Management

### Synthetic Data

Use synthetic data with known properties:
- XOR: Simple, well-understood
- Linear: y = ax + b
- Nonlinear: y = sin(x)

### Edge Cases

Test with edge cases:
- Zero inputs
- Negative inputs
- Large values
- Small values
- Identical values

### Random Data

Use fixed random seeds for reproducibility:
```python
import numpy as np
np.random.seed(42)
```

---

## Test Reporting

### Test Results

Record test results:
- Total tests
- Passed tests
- Failed tests
- Skipped tests
- Coverage percentage

### Failure Investigation

For test failures:
1. Identify failing test
2. Reproduce failure
3. Identify root cause
4. Fix issue
5. Re-run tests
6. Document fix

---

## Testing Tools

| Tool | Purpose |
|------|---------|
| pytest | Test framework |
| pytest-cov | Coverage reporting |
| numpy.testing | Numerical assertions |

### pytest Configuration

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
```

---

## Quality Gates

Before moving to next stage:

| Gate | Requirement |
|------|-------------|
| Unit tests | All unit tests pass |
| Gradient checking | All gradients within tolerance |
| Integration tests | Training flow works |
| No regressions | No previously passing tests fail |

---

## Open Questions

- Should we add property-based testing?
- Should we add fuzz testing?
- Should we add performance tests?
- Should we add memory tests?

These are future considerations, not current requirements.
