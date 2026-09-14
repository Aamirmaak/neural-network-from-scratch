# 02 — Technical Requirements Document

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 0 — Planning

## Language

**Python 3.8+**

The project uses Python as the implementation language. Type hints should be used where they improve clarity without adding unnecessary complexity.

## Numerical Dependency

**NumPy** may be used for numerical operations.

NumPy provides efficient array operations and is the standard numerical library in Python. Using NumPy for low-level array operations is acceptable; using NumPy's automatic differentiation or gradient-tracking features is not.

## Explicit Restrictions

The core implementation must NOT use:

| Library | Reason |
|---------|--------|
| PyTorch | Deep-learning framework |
| TensorFlow | Deep-learning framework |
| JAX | Automatic-differentiation framework |
| Keras | Deep-learning API |
| autograd | Automatic-differentiation library |
| jax.autograd | Automatic-differentiation module |
| torch.autograd | Automatic-differentiation module |
| Any autodiff library | Purpose is to implement learning machinery ourselves |

**Rationale:** The purpose of this project is to implement the fundamental machinery of neural-network training from first principles. Using existing autodiff or deep-learning libraries would defeat this purpose.

**Exception:** Libraries may be used for:

- Visualization (matplotlib)
- Testing (pytest)
- Project tooling (packaging, linting)
- Experiment analysis (pandas, seaborn)

These must be clearly distinguished from the core implementation.

## Engineering Principles

### Core Principles

| Principle | Description | Application |
|-----------|-------------|-------------|
| Correctness | Code must produce correct results | Validate with tests and gradient checking |
| Readability | Code must be easy to understand | Clear naming, minimal abstraction |
| Modularity | Components must have clear interfaces | Separate concerns, dependency injection |
| Testability | Code must be easy to test | Pure functions, minimal side effects |
| Reproducibility | Results must be reproducible | Fixed seeds, deterministic operations |
| Explainability | Design decisions must be documented | Architecture docs, decision log |

### Code Quality

| Requirement | Description |
|-------------|-------------|
| Type hints | Use where they improve clarity |
| Docstrings | Document public interfaces |
| Small files | Avoid files larger than ~300 lines |
| Minimal abstraction | No abstraction until needed |
| Clear naming | Variables and functions should be self-documenting |
| Consistent style | Follow PEP 8 |

### Architecture

| Requirement | Description |
|-------------|-------------|
| Single responsibility | Each module has one clear purpose |
| Clear interfaces | Public APIs are well-defined |
| Dependency direction | Dependencies flow downward |
| No circular imports | Modules may not depend on each other cyclically |

## Environment

### Python Version

Python 3.8 or later. The implementation should use features available in Python 3.8+ without requiring newer versions.

### Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| numpy | Numerical operations | 1.20+ |
| matplotlib | Visualization (optional) | 3.0+ |
| pytest | Testing (optional) | 6.0+ |

Dependency versions should eventually be pinned in a `requirements.txt` file for reproducibility.

### Development Tools

| Tool | Purpose |
|------|---------|
| pytest | Testing |
| black | Code formatting (optional) |
| flake8 | Linting (optional) |
| mypy | Type checking (optional) |

## Quality Requirements

### Determinism

| Requirement | Description |
|-------------|-------------|
| Random seeds | All random operations must accept a seed |
| Seed documentation | Seeds must be recorded in experiment configurations |
| Deterministic mode | When seeds are fixed, results must be deterministic |

### Testing

| Requirement | Description |
|-------------|-------------|
| Unit tests | All operations must have unit tests |
| Gradient checking | All gradients must be verified numerically |
| Integration tests | Training flows must be tested |
| Coverage | Aim for high coverage of core components |

### Numerical Correctness

| Requirement | Description |
|-------------|-------------|
| Tolerance | Gradient checking uses appropriate tolerance |
| Edge cases | Handle zeros, infinities, NaN appropriately |
| Stability | Implementations must be numerically stable |

### Experiment Reproducibility

| Requirement | Description |
|-------------|-------------|
| Configuration | Experiments are defined by configuration files |
| Seed recording | Seeds are recorded with results |
| Result storage | Results are stored with full configuration |
| Documentation | Experiments are fully documented |

### Documentation

| Requirement | Description |
|-------------|-------------|
| Consistency | Terminology is consistent across documents |
| Accuracy | Documentation matches implementation |
| Currency | Documentation is updated with changes |

## Performance Constraints

Performance is NOT a primary concern. The project prioritizes:

1. Correctness
2. Readability
3. Educational value

Performance optimizations may be considered only after correctness is established, and only if they do not compromise readability.

## Security Constraints

See [09_SECURITY.md](09_SECURITY.md) for security considerations.

## Compliance

| Requirement | Description |
|-------------|-------------|
| License | Choose an appropriate open-source license |
| Attribution | Document any external code or ideas |
| Academic honesty | Do not claim others' work as original |

## Open Questions

None at this stage.

## Appendix: Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python | Industry standard for ML, high readability |
| Numerical | NumPy | Standard, efficient, well-understood |
| Testing | pytest | Python standard, good support |
| Visualization | matplotlib | Standard, sufficient for educational use |
| Formatting | PEP 8 | Python standard style |
