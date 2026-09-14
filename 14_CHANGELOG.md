# 14 — Changelog

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 2 In Progress

## Overview

This document records changes to the project.

**Format:** Follows [Keep a Changelog](https://keepachangelog.com/) format.

---

## [Unreleased]

### Added

- Stage 2: Extended autodiff operations in `src/neuralearn/value.py`:
  - Division (`__truediv__`, `__rtruediv__`) with quotient rule backward
  - Reciprocal (`reciprocal()`) with -1/x² backward
  - Exponential (`exp()`) with exp(x) backward
  - Natural logarithm (`log()`) with 1/x backward (domain: x > 0)
  - Hyperbolic tangent (`tanh()`) with (1-tanh²) backward
  - ReLU (`relu()`) with gradient=0 at x=0 convention
  - `import math` for exp/log/tanh
- 65 new tests for Stage 2 operations (152 total):
  - Division, reciprocal, exp, log, tanh, relu: forward and backward
  - Numerical gradient checks for all new operations
  - Scalar interoperability for division
  - Chained expressions mixing Stage 1 and Stage 2 operations
  - Repeated backward with new operations (D19 semantics)
- Updated documentation:
  - `03_PROJECT_PLAN.md` — Stage 2 defined, current stage updated
  - `04_ARCHITECTURE.md` — Stage 2 operations table with derivatives
  - `10_DECISIONS.md` — D20-D24 (division, log domain, ReLU at zero, reciprocal, single file)
  - `11_PROGRESS_LOG.md` — Entry 003 (Stage 2 work, 152 tests)
  - `13_LEARNINGS.md` — 2 new learning entries (quotient rule, ReLU at zero)
  - `14_CHANGELOG.md` — This entry

### Changed

- Version bumped from 0.2.0 to 0.3.0

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

### Stage 3 (Planned)

- Gradient checking as separate utility module
- Additional gradient verification

### Stages 4-14 (Planned)

- See 03_PROJECT_PLAN.md for full roadmap
