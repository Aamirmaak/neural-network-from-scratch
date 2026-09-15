# 11 — Progress Log

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 3 COMPLETE

## Overview

This document tracks progress through the project stages.

**Format:** Each entry records Date, Stage, Work Completed, What Was Learned, Problems, and Next Step.

---

## Entry 001

**Date:** 2026-09-15  
**Stage:** Stage 0 — Project Definition & Documentation

### Work Completed

- Initialized Git repository
- Created project directory structure
- Created .gitignore
- Created src/neuralearn/__init__.py
- Created README.md
- Created 01_PRD.md (Product Requirements Document)
- Created 02_TRD.md (Technical Requirements Document)
- Created 03_PROJECT_PLAN.md (Project Plan)
- Created 04_ARCHITECTURE.md (Architecture)
- Created 05_RESEARCH_PLAN.md (Research Plan)
- Created 06_EXPERIMENT_PLAN.md (Experiment Plan)
- Created 07_TESTING_STRATEGY.md (Testing Strategy)
- Created 08_DEPLOYMENT_PLAN.md (Deployment Plan)
- Created 09_SECURITY.md (Security)
- Created 10_DECISIONS.md (Architectural Decision Log)
- Created 11_PROGRESS_LOG.md (this file)
- Created 12_EXPERIMENT_LOG.md (Experiment Log)
- Created 13_LEARNINGS.md (Learning Log)
- Created 14_CHANGELOG.md (Changelog)

### What Was Learned

- Documentation-first approach forces clear thinking about design
- Project scope is well-defined through documentation
- Architecture decisions are documented and justified
- Testing strategy is comprehensive

### Problems

- None at this stage

### Next Step

- Next authorized stage: Stage 1 — Core Value / Tensor Abstraction
- Stage 1 has NOT begun. It will be authorized separately.

---

## Template for Future Entries

```markdown
## Entry [NNN]

**Date:** [YYYY-MM-DD]  
**Stage:** [Stage Name]

### Work Completed

- [List completed work items]

### What Was Learned

- [List learning points]

### Problems

- [List problems encountered, or "None"]

### Next Step

- [Describe next steps]
```

---

## Entry 002

**Date:** 2026-09-15  
**Stage:** Stage 1 — Core Value / Computational Graph

### Work Completed

- Implemented `Value` class in `src/neuralearn/value.py` (~250 lines)
  - Scalar data storage with float conversion
  - Gradient storage (initialized to 0.0)
  - Parent node tracking (`_prev` set)
  - Operation metadata (`_op` string)
  - Local backward function (`_backward` callable)
- Implemented 5 forward operations:
  - Addition (`+`, `__add__`, `__radd__`)
  - Multiplication (`*`, `__mul__`, `__rmul__`)
  - Negation (`-x`, `__neg__`)
  - Subtraction (`-`, `__sub__`, `__rsub__`)
  - Scalar power (`**`, `__pow__`)
- Implemented reverse-mode backward propagation:
  - Root gradient initialization (d(output)/d(output) = 1)
  - Topological sort via DFS
  - Reverse topological traversal
  - Local backward rules for each operation
  - Gradient accumulation via `+=` for shared nodes
  - Intermediate gradient reset for correct repeated-backward semantics (D19)
- Implemented scalar interoperability for all operations
- Updated `__init__.py` to export `Value`, version 0.2.0
- Wrote 87 comprehensive tests in `tests/test_value.py`:
  - 6 construction/representation tests
  - 7 basic operation forward-value tests
  - 10 backward propagation tests (per operation)
  - 2 root gradient tests
  - 4 chained expression tests
  - 5 branching/shared-node tests
  - 10 scalar interoperability tests
  - 11 numerical gradient checking tests (finite differences)
  - 11 edge case tests (including 4 repeated-backward tests with intermediate nodes)
  - 2 topological ordering tests
  - 5 integration-style tests
  - 4 regression tests
- All 87 tests pass
- Updated 7 documentation files (03, 04, 07, 10, 11, 13, 14)

### Implementation Decisions

- D16: Internal naming convention (`_prev`, `_op`, `_backward`) for graph fields
- D17: All Stage 1 operations in single `value.py` module (~250 lines)
- D18: Numerical gradient checking in tests (not separate utility module)
- D19: Repeated backward semantics — leaf accumulation, intermediate reset

### Tests Performed

- 87 total tests, 87 passed, 0 failed
- Numerical gradient verification for all 5 operations + chained expressions
- Branching graph gradient accumulation (x*x + x, x*x*x, shared nodes)
- Edge cases: zeros, negatives, large/small values, deep graphs
- Repeated backward with intermediate nodes, branching graphs, 3-call accumulation

### Actual Learning

- Debugging revealed that numerical gradient tests must use closures capturing Value objects (not create new Values inside `fn()`), otherwise perturbation has no effect
- Loop variable rebinding (`a = a * 2.0`) works correctly for graph construction because Python keeps the old objects alive via `_prev` references
- The output node's gradient is always 1.0 by definition; leaf node gradients are what accumulate through the graph
- **Repeated backward bug:** Initial implementation only reset root.grad, leaving stale intermediate gradients. This caused incorrect leaf gradients on second backward call (96.0 instead of 64.0 for `z = (x*x)^2`). Fix: reset non-leaf, non-root nodes to 0 before each traversal.

### Problems

- Initial numerical gradient tests failed because `fn()` created new Value objects each call
- `test_deep_mul_chain` initially asserted output node grad instead of leaf node grad
- Repeated backward with intermediate nodes produced incorrect gradients (fixed by D19)

### Next Step

- Stage 1 is complete pending architect review
- Next authorized stage: Stage 2 — Automatic Differentiation (Extended)

---

## Entry 003

**Date:** 2026-09-15  
**Stage:** Stage 2 — Extended Autodiff Operations

### Work Completed

- Extended `src/neuralearn/value.py` (~405 lines total)
  - Added `import math` for exp/log/tanh
  - Added `__truediv__` and `__rtruediv__` — division with backward rule
  - Added `exp()` — exponential with backward rule
  - Added `log()` — natural log with backward rule (domain: x > 0)
  - Added `tanh()` — hyperbolic tangent with backward rule
  - Added `relu()` — rectified linear unit with backward rule (grad=0 at x=0)
  - Added `reciprocal()` — convenience method for 1/x
- Added 65 new tests to `tests/test_value.py` (152 total):
  - Division: 9 tests (6 forward, 3 backward)
  - Reciprocal: 5 tests (3 forward, 2 backward)
  - Exp: 6 tests (4 forward, 2 backward)
  - Log: 7 tests (4 forward, 3 backward)
  - Tanh: 7 tests (4 forward, 3 backward)
  - ReLU: 9 tests (4 forward, 5 backward)
  - Numerical gradient checks: 9 tests (single ops, chains, mixed Stage 1+2)
  - Scalar interoperability: 4 tests
  - Chained expressions: 4 tests
  - Repeated backward: 5 tests
- All 152 tests pass (87 Stage 1 + 65 Stage 2)
- Manual mathematical validation passed for all operations
- Scope audit: no forbidden operations (layers, losses, optimizers, training) leaked in
- Updated documentation: 03, 04, 10 (decisions D20-D24)

### Implementation Decisions

- D20: Division as direct operation (not composed as x * y**-1)
- D21: Log domain restricted to x > 0 (no explicit check, caller's responsibility)
- D22: ReLU at zero convention — gradient = 0 (standard in deep learning)
- D23: Reciprocal as convenience method (backward: -1/x²)
- D24: All new operations in value.py (single file, consistent with D17)

### Actual Learning

- Division backward uses the quotient rule: ∂(x/y)/∂x = 1/y, ∂(x/y)/∂y = -x/y²
- The `numerical_grad` helper takes a no-arg callable and perturbs the Value's data directly — tests must use closures capturing Value objects
- ReLU's non-smoothness at x=0 requires an explicit convention; gradient=0 is standard
- `exp(log(x)) = x` is a useful identity for testing — gradient should be 1
- Tanh backward: `∂tanh/∂x = 1 - tanh(x)²` — simple but requires caching the forward value

### Problems

- Initial reciprocal tests failed because `reciprocal()` method was not added to Value class
- Numerical gradient tests initially used `fn(a)` calling convention instead of no-arg closures

### Next Step

- Stage 2 implementation complete, pending acceptance
- Next authorized stage: Stage 3 — Gradient Checking

---

## Entry 004

**Date:** 2026-09-15  
**Stage:** Stage 3 — Gradient Checking

### Work Completed

- Created `src/neuralearn/gradient_check.py` (~120 lines)
  - `numerical_grad(fn, x_val, eps)` — central-difference gradient for single Value
  - `gradient_check(fn, values, eps, atol, rtol)` — multi-value gradient comparison
  - Returns structured results with analytical, numerical, abs_error, rel_error, passed
- Created `tests/test_gradient_check.py` (43 tests)
  - numerical_grad helper tests (4)
  - gradient_check utility tests (4)
  - Individual operation gradient checks: add, mul, sub, pow, div, reciprocal, exp, log, tanh, relu (15)
  - Composed graph tests (4)
  - Multi-input tests with 3 and 4 inputs (2)
  - Deep graph tests: chain, mixed ops, Stage 2 ops (3)
  - Numerical edge cases: near-zero div, large exp, saturated tanh, small log (4)
  - Regression tests: gradient corruption detection, D19 compatibility (4)
- Updated `tests/test_value.py` to import `numerical_grad` from module
- Updated 6 documentation files (03, 07, 10, 11, 13, 14)
- All 195 tests pass (152 existing + 43 new)

### Implementation Decisions

- D25: Gradient checking as separate reusable module
- D26: ε=1e-5, atol=1e-5, rtol=1e-3

### Actual Learning

- Central difference O(ε²) truncation error is more accurate than forward difference O(ε)
- ε=1e-5 balances truncation (~1e-10) against roundoff (~1e-11)
- ReLU at x=0 is nondifferentiable; numerical gradient (~0.5) differs from convention (0) — document, don't force match
- gradient_check must not call backward() — it reads existing .grad values to avoid graph corruption
- Detecting wrong gradients: setting a.grad=99.0 should fail the check — verified

### Problems

- None encountered during implementation

### Next Step

- Stage 3 implementation complete, pending acceptance
- Next authorized stage: Stage 4 — Parameters and Layers

---

## Stage Progress

| Stage | Status | Start Date | End Date |
|-------|--------|------------|----------|
| Stage 0 | COMPLETE | 2026-09-15 | 2026-09-15 |
| Stage 1 | COMPLETE (approved) | 2026-09-15 | 2026-09-15 |
| Stage 2 | COMPLETE (approved) | 2026-09-15 | 2026-09-15 |
| Stage 3 | IN PROGRESS | 2026-09-15 | 2026-09-15 |
| Stage 4 | NOT STARTED | - | - |
| Stage 5 | NOT STARTED | - | - |
| Stage 6 | NOT STARTED | - | - |
| Stage 7 | NOT STARTED | - | - |
| Stage 8 | NOT STARTED | - | - |
| Stage 9 | NOT STARTED | - | - |
| Stage 10 | NOT STARTED | - | - |
| Stage 11 | NOT STARTED | - | - |
| Stage 12 | NOT STARTED | - | - |
| Stage 13 | NOT STARTED | - | - |
| Stage 14 | NOT STARTED | - | - |

---

## Metrics

| Metric | Value |
|--------|-------|
| Current Stage | 3 |
| Documentation Files | 15 |
| Source Files | 3 |
| Test Files | 2 |
| Experiment Files | 0 |
| Total Tests | 195 |
| Tests Passed | 195 |
