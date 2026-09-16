# 14 — Changelog

**Project:** Neural Network From Scratch  
**Version:** 1.3  
**Status:** Stage 12 COMPLETE

## Overview

This document records changes to the project.

**Format:** Follows [Keep a Changelog](https://keepachangelog.com/) format.

---

## [1.3.0] — Stage 12 — Package Distribution + CLI

### Added

- `src/neuralearn/cli.py` — lightweight CLI with `--version`, `info`, `example` commands
- `console_scripts` entry point: `neuralearn` command available after install
- `tests/test_cli.py` — 12 CLI tests (parser + subprocess)
- Build artifacts: source distribution and wheel

### Fixed

- `pyproject.toml` build backend corrected from `setuptools.backends._legacy:_Backend` to `setuptools.build_meta`

---

## [1.2.0] — Stage 11 — Framework/API Polish

### Added

- `pyproject.toml` — proper package installation with optional `[viz]` and `[dev]` extras
- `gradient_check` and `numerical_grad` exported from package root
- `Value.__rpow__` — support `2 ** Value(3)` syntax
- `tests/test_api.py` — 7 API smoke tests (clean imports, full training workflows)
- README quickstart example

### Fixed

- `Module.forward()` return type annotation (was `Value`, now untyped for flexibility)
- Visualization functions now lazy-loaded from `__init__.py` (no matplotlib import at package level)
- `visualization.py` removed dead code (unused `plot_train_test_loss` alias, unused numpy meshgrid)
- `visualization.py` added missing type hints (`Callable` for `target_fn`, `Any` for `history`)
- `datasets.py` removed unused `List` import
- README updated: proper quickstart, `pyproject.toml` in structure, correct API examples

---

## [1.1.0] — Stage 10 — Visualization & Experiment Analysis
- Stage 7: Training engine:
  - `src/neuralearn/training.py` — Trainer class with fit/evaluate API
  - `tests/test_training.py` — 40 training engine tests
- Stage 6: Optimizers:
  - `src/neuralearn/optimizers.py` — SGD, MomentumSGD, Adam
  - `tests/test_optimizers.py` — 44 optimizer tests
- Stage 5: Loss functions:
  - `src/neuralearn/losses.py` — mse_loss, binary_cross_entropy
  - `tests/test_losses.py` — 58 loss tests including gradient checking
- Stage 4: Parameters and layers:
  - `src/neuralearn/parameter.py` — Parameter class (subclasses Value, zero_grad, requires_grad)
  - `src/neuralearn/layers.py` — Module, Neuron, Linear, ReLU, Tanh
  - `tests/test_parameter.py` — 15 parameter tests
  - `tests/test_layers.py` — 42 layer tests including gradient checking
- Updated documentation files (03, 04, 07, 10, 11, 13, 14)

### Changed

- Version bumped from 1.0.0 to 1.1.0

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
| 1.1.0 | 2026-09-16 | 10 | Visualization module, plot generation, 23 tests, D49-D51, 5 generated plots |
| 1.0.0 | 2026-09-16 | 9 | End-to-end experiments (XOR, regression), Sigmoid layer, 465 tests, D48 |
| 0.9.0 | 2026-09-16 | 8 | Dataset, DataLoader, batching/shuffling/seed, Trainer integration, 63 tests, D43-D47 |
| 0.8.0 | 2026-09-16 | 7 | Trainer class with fit/evaluate, per-sample lifecycle, history, 40 tests, D38-D42 |
| 0.7.0 | 2026-09-15 | 6 | SGD, MomentumSGD, Adam optimizers, 44 tests, D34-D37 |
| 0.6.0 | 2026-09-15 | 5 | MSE, BCE losses, 58 tests, D31-D33 |
| 0.5.0 | 2026-09-15 | 4 | Parameter, Module, Neuron, Linear, ReLU, Tanh, 57 tests, D27-D30 |
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
