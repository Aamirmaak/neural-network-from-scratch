# Contributing to NeuraLearn

Thank you for your interest in contributing to NeuraLearn! This document provides guidelines for contributing to this educational neural-network framework.

## Project Purpose

NeuraLearn is an educational framework that implements neural-network training from first principles. The goal is to demonstrate understanding of automatic differentiation, backpropagation, and deep-learning fundamentals without relying on existing ML frameworks.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Aamirmaak/neural-network-from-scratch.git
cd neural-network-from-scratch

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install in development mode
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -q

# Run with verbose output
python -m pytest tests/ -v
```

All tests must pass before submitting a pull request.

## Coding Expectations

- **Correctness first:** Every operation must be mathematically correct
- **Tests required:** All new functionality must include tests
- **Documentation:** Public APIs must have docstrings
- **Simplicity:** Prefer simple, clear code over clever implementations
- **No external ML frameworks:** Do not use PyTorch, TensorFlow, JAX, or similar

## Documentation Expectations

- Update README.md if adding user-facing features
- Update relevant documentation files (03-14) for architectural changes
- Keep changelog entries concise and accurate

## Test Expectations

- Write tests that verify behavior, not implementation details
- Include edge cases where relevant
- Use numerical gradient checking for new autodiff operations
- Ensure tests are deterministic where possible

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `master`
3. Make your changes with tests
4. Ensure all tests pass
5. Update documentation as needed
6. Submit a pull request with a clear description

## Commit Messages

Use clear, descriptive commit messages:
- `feat: add new feature`
- `fix: resolve bug in X`
- `docs: update documentation`
- `test: add tests for Y`
- `refactor: improve code structure`

## Scope Boundaries

This project is intentionally limited to:
- Scalar reverse-mode autodiff
- Basic neural-network layers (Linear, ReLU, Tanh, Sigmoid)
- Standard losses (MSE, BCE)
- Standard optimizers (SGD, Momentum, Adam)
- Training infrastructure
- Visualization

We do NOT accept:
- Tensor/GPU support
- CNN, RNN, Transformer implementations
- Cloud deployment infrastructure
- Advanced production features

## Code of Conduct

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

## Questions?

Open an issue with the `question` label if you have questions about contributing.
