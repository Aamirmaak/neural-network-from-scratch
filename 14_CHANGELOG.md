# 14 — Changelog

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 3 COMPLETE

## Overview

This document records changes to the project.

**Format:** Follows [Keep a Changelog](https://keepachangelog.com/) format.

---

## [Unreleased]

### Added

- Stage 3: Gradient-checking subsystem:
  - `src/neuralearn/gradient_check.py` — reusable module with `numerical_grad` and `gradient_check` functions
  - `tests/test_gradient_check.py` — 43 comprehensive gradient-checking tests
  - Central-difference finite-difference approximation (ε=1e-5)
  - Multi-value gradient comparison with absolute and relative tolerance
  - Tests covering all individual operations, composed graphs, multi-input, deep graphs
  - Numerical edge case documentation (ReLU at zero, near-zero division, saturated tanh)
  - Regression tests for D19 compatibility and incorrect gradient detection
- Updated `tests/test_value.py` to import `numerical_grad` from gradient_check module
- Updated 6 documentation files (03, 07, 10, 11, 13, 14)

### Changed

- Version bumped from 0.3.0 to 0.4.0

### Deprecated

- Nothing yet

### Removed

- Nothing yet

### Fixed

- Nothing yet

### Security

- Nothing yet

---

## [0.1.0] — 2026-09-15

### Added

- Initial project documentation
- Project structure
- All required documentation files

---

## Version History

| Version | Date | Stage | Description |
|---------|------|-------|-------------|
| 0.4.0 | 2026-09-15 | 3 | Gradient checking module: numerical_grad, gradient_check, 43 tests, D25-D26 |
| 0.3.0 | 2026-09-15 | 2 | Extended ops: div, reciprocal, exp, log, tanh, relu, 152 tests, D20-D24 |
| 0.2.0 | 2026-09-15 | 1 | Value class, forward ops, backward propagation, 87 tests, D19 backward fix |
| 0.1.0 | 2026-09-15 | 0 | Initial documentation and scaffolding |

---

## Change Categories

- **Added:** New features
- **Changed:** Changes to existing functionality
- **Deprecated:** Features that will be removed
- **Removed:** Features that have been removed
- **Fixed:** Bug fixes
- **Security:** Vulnerability fixes

---

## Future Changes

As the project progresses, changes will be recorded here.

### Stage 4 (Planned)

- Trainable parameters
- Linear, ReLU, Tanh layers
- Sequential container

### Stages 5-14 (Planned)

- See 03_PROJECT_PLAN.md for full roadmap
