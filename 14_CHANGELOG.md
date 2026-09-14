# 14 — Changelog

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 1 Complete — Awaiting Architect Review

## Overview

This document records changes to the project.

**Format:** Follows [Keep a Changelog](https://keepachangelog.com/) format.

---

## [Unreleased]

### Added

- `src/neuralearn/value.py` — Scalar Value class with reverse-mode autodiff
  - Value class with data, grad, _prev, _op, _backward
  - Addition (+), multiplication (*), negation (-x), subtraction (-), power (**)
  - Scalar interoperability (Value + scalar, scalar * Value, etc.)
  - Backward propagation with topological sort and gradient accumulation
  - Intermediate gradient reset for correct repeated-backward semantics (D19)
- `tests/test_value.py` — 87 comprehensive tests
  - Construction, forward operations, backward propagation
  - Chained expressions, branching graphs, scalar interoperability
  - Numerical gradient checking (finite differences)
  - Edge cases, topological ordering, integration, regression tests
  - Repeated backward with intermediate nodes (3 new tests)
- Updated `src/neuralearn/__init__.py` — exports Value, version 0.2.0
- Updated `03_PROJECT_PLAN.md` — Stage 1 status COMPLETE, scalar-only scope note
- Updated `04_ARCHITECTURE.md` — Value, computational graph, backward semantics, power domain
- Updated `07_TESTING_STRATEGY.md` — Stage 1 test cases including repeated backward
- Updated `10_DECISIONS.md` — Stage 1 decisions (D16-D19)
- Updated `11_PROGRESS_LOG.md` — Entry 002 (Stage 1 work, 87 tests)
- Updated `13_LEARNINGS.md` — 4 verified learning entries from Stage 1
- Updated `14_CHANGELOG.md` — This entry

### Changed

- Version bumped from 0.1.0 to 0.2.0

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

### Stage 2 (Planned)

- Extended autodiff operations: division, exp, log, tanh, ReLU
- Corresponding backward rules
- Additional gradient checking

### Stages 3-14 (Planned)

- See 03_PROJECT_PLAN.md for full roadmap
