# 10 — Architectural Decision Log

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 7 IN PROGRESS

## Overview

This document records architectural decisions made during the project.

**Format:** Each decision includes Context, Reasoning, Consequences, and Status.

---

## D01: Python as Implementation Language

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Use Python as the implementation language.

### Context

The project requires implementing a neural-network framework from scratch. The language must support numerical computing, be widely understood, and be suitable for educational purposes.

### Reasoning

- Python is the industry standard for ML/AI
- High readability supports educational goals
- Large ecosystem for numerical computing
- Easy to find developers who can read/extend
- Good documentation and community support

### Consequences

- Access to NumPy for numerical operations
- Python's performance limitations (acceptable for educational scope)
- Dynamic typing (mitigated with type hints)
- Global interpreter lock (acceptable for single-device training)

---

## D02: NumPy as Allowed Numerical Dependency

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

NumPy may be used for numerical operations in the core implementation.

### Context

The project needs efficient numerical operations. Implementing array operations from scratch would be excessive and not educational.

### Reasoning

- NumPy provides efficient array operations
- Using NumPy for low-level operations is acceptable
- NumPy's autodiff features are NOT used
- Focus is on implementing learning machinery, not array operations

### Consequences

- Efficient numerical operations
- Familiar API for ML practitioners
- Clear boundary: NumPy for arrays, custom code for autodiff
- Must not use NumPy's gradient-tracking features

---

## D03: Exclusion of Deep-Learning Frameworks

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

The core implementation must NOT use PyTorch, TensorFlow, JAX, Keras, or automatic-differentiation libraries.

### Context

The purpose is to implement learning machinery from first principles. Using existing frameworks would defeat this purpose.

### Reasoning

- Educational value requires implementing core mechanics
- Understanding autodiff is a primary learning objective
- Demonstrates deep understanding for portfolio
- Enables debugging and extending from first principles

### Consequences

- More implementation work
- Deeper understanding of internals
- Clear demonstration of competence
- Limitations in scope (acceptable)

---

## D04: Documentation-First Workflow

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Create documentation before implementation.

### Context

The project needs clear requirements, design, and planning before coding begins.

### Reasoning

- Prevents scope creep
- Forces clear thinking about design
- Creates reference for implementation
- Demonstrates engineering methodology

### Consequences

- More upfront work
- Clearer implementation path
- Better documentation
- Easier to track progress

---

## D05: Modular Project Structure

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Use a modular project structure with separate directories for source, tests, experiments, configs, and scripts.

### Context

The project needs organization that supports development, testing, and experimentation.

### Reasoning

- Clear separation of concerns
- Easy to find components
- Supports independent testing
- Enables parallel development

### Consequences

- Clear file organization
- Easy navigation
- Standard Python project layout
- Supports tooling (pytest, etc.)

---

## D06: Separation of Source, Tests, Experiments

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Keep source code, tests, and experiments in separate directories.

### Context

Different types of code have different purposes and different dependencies.

### Reasoning

- Library code is reusable
- Tests verify correctness
- Experiments are specific investigations
- Different dependencies (e.g., matplotlib for experiments)

### Consequences

- Clean library code
- Focused test code
- Independent experiment code
- Clear dependency boundaries

---

## D07: Educational Correctness as Priority

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Prioritize educational correctness and explainability over performance.

### Context

The project is educational, not production. Understanding is more valuable than speed.

### Reasoning

- Primary goal is learning
- Code should be readable and understandable
- Performance optimizations can obscure logic
- Correctness is more valuable than speed

### Consequences

- Slower execution (acceptable)
- Clearer code
- Better documentation
- Easier to extend

---

## D08: Staged Development

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Implement in stages with validation at each stage.

### Context

The project is complex and needs incremental development with verification.

### Reasoning

- Reduces risk of large failures
- Enables validation at each step
- Creates clear progress markers
- Supports debugging

### Consequences

- Clear milestones
- Easier debugging
- Better documentation
- More predictable timeline

---

## D09: Value Class as Core Abstraction

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Implement a Value class that wraps numerical data with gradient tracking.

### Context

The framework needs a numerical abstraction that supports automatic differentiation.

### Reasoning

- Value class is foundation for computational graph
- Stores metadata needed for backpropagation
- Clear and simple design
- Matches how real frameworks work

### Consequences

- All operations create new Values
- Metadata enables autodiff
- Clear data flow
- Easy to inspect and debug

---

## D10: Reverse-Mode Automatic Differentiation

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Implement reverse-mode (backpropagation) rather than forward-mode autodiff.

### Context

The framework needs automatic differentiation for training neural networks.

### Reasoning

- Reverse-mode is efficient for scalar outputs (loss)
- This is how real frameworks work
- Educational value: understanding backpropagation
- Standard approach in deep learning

### Consequences

- Efficient gradient computation
- Requires topological ordering
- Requires storing intermediate values
- Standard and well-understood

---

## D11: Topological Ordering for Backward Pass

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Use topological sorting for backward pass ordering.

### Context

Reverse-mode autodiff requires computing gradients in correct dependency order.

### Reasoning

- Ensures gradients computed in correct order
- Required for correct chain rule application
- Prevents using gradients before computed
- Standard approach

### Consequences

- Correct gradient computation
- Clear implementation
- Well-understood algorithm
- Handles complex graphs

---

## D12: Layers Separate from Autodiff Core

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Keep layers as a separate module from the autodiff core.

### Context

Layers are specific to neural networks; autodiff is generic.

### Reasoning

- Clear separation of concerns
- Autodiff core is reusable
- Layers are specific implementations
- Easier to test independently

### Consequences

- Clean module boundaries
- Easier testing
- Better organization
- More extensible

---

## D13: Optimizers Independent from Models

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Optimizers are separate objects that take parameters as input.

### Context

Different optimizers can be used with same model.

### Reasoning

- Clear interface
- Different optimizers, same model
- Matches how real frameworks work
- Easy to swap optimizers

### Consequences

- Flexible optimizer selection
- Clean interface
- Easy experimentation
- Standard approach

---

## D14: Experiments Separate from Library Code

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Experiments live in a separate directory, not in the library.

### Context

Experiments are specific investigations; library code is reusable.

### Reasoning

- Library code is clean and reusable
- Experiments may have different dependencies
- Keeps library focused
- Enables multiple experiments

### Consequences

- Clean library code
- Independent experiments
- Clear organization
- Easy to add new experiments

---

## D15: Package Name "neuralearn"

**Date:** Stage 0  
**Status:** ACCEPTED

### Decision

Use "neuralearn" as the Python package name.

### Context

The project needs a package name for the source code.

### Reasoning

- Short and memorable
- Descriptive of purpose
- No conflicts with existing packages
- Easy to type and pronounce

### Consequences

- `import neuralearn`
- Consistent naming throughout
- Professional appearance

---

## Stage 2 Decisions

### D20: Division Implemented as Direct Operation

**Date:** Stage 2  
**Status:** ACCEPTED

### Decision

Implement division as a direct `__truediv__` operation rather than composing as `x * (y ** -1)`.

### Context

Division is a common operation that benefits from a dedicated backward rule rather than relying on composed power/multiplication derivatives.

### Reasoning

- Cleaner forward computation (single node in graph vs. two nodes)
- More efficient backward pass (fewer gradient computations)
- The backward rule `∂z/∂x = 1/y`, `∂z/∂y = -x/y²` is straightforward
- Avoids the `y ** -1` power rule complexity for a simple division

### Consequences

- Cleaner graph structure for division operations
- More efficient gradient computation
- Simpler to understand and debug

---

### D21: Log Domain Restriction (x > 0)

**Date:** Stage 2  
**Status:** ACCEPTED

### Decision

Log is defined only for positive inputs (x > 0). No explicit domain check is enforced; the caller is responsible.

### Context

Log is undefined for x ≤ 0 (complex values or undefined). The autodiff engine operates on real numbers.

### Reasoning

- Matches mathematical definition of log(x)
- No runtime domain check avoids overhead in the hot path
- Caller's responsibility (consistent with power operation in Stage 1)
- Documenting the domain expectation is sufficient

### Consequences

- Log is undefined for x ≤ 0
- Caller must ensure valid inputs
- Documented domain constraint

---

### D22: ReLU at Zero Convention (gradient = 0)

**Date:** Stage 2  
**Status:** ACCEPTED

### Decision

For ReLU at x = 0, define the gradient as 0.

### Context

ReLU is non-differentiable at x = 0. We must choose a convention for the gradient at this point.

### Reasoning

- Standard convention in deep learning frameworks
- `∂ReLU/∂x = 0` at x = 0 means the gradient does not flow backward through the zero point
- This is consistent with the "dead neuron" interpretation
- Numerically stable and simple to implement

### Consequences

- ReLU gradient at x = 0 is 0
- No special case handling needed in the backward pass
- Standard behavior matches PyTorch/TensorFlow

---

### D23: Reciprocal as Convenience Operation

**Date:** Stage 2  
**Status:** ACCEPTED

### Decision

Implement `reciprocal()` as a convenience method (equivalent to `1 / x`).

### Context

Reciprocal is useful in normalization and is mathematically simple. Having a dedicated method avoids repeated `(x ** -1)` or `(1 / x)` composition.

### Reasoning

- Common operation in normalization
- Simple backward rule: `∂z/∂x = -1/x²`
- Convenience method for cleaner user code
- Can be composed as `1 / x` but dedicated method is clearer

### Consequences

- Convenience method available
- Internally uses the same backward rule as division by a constant

---

### D24: All New Operations in value.py

**Date:** Stage 2  
**Status:** ACCEPTED

### Decision

Continue implementing all new operations (division, reciprocal, exp, log, tanh, ReLU) in `value.py` rather than splitting into a separate operations module.

### Context

The file remains manageable (~350-400 lines expected). Splitting would add indirection without benefit at this scale.

### Reasoning

- Keeps related code together
- Easy to read and understand the complete system
- Consistent with D17 from Stage 1
- Can be split later if the file grows too large

### Consequences

- Single file contains all operations
- Easy to navigate and understand
- May need splitting in later stages (acceptable)

---

## Stage 3 Decisions

### D25: Gradient Checking as Separate Module

**Date:** Stage 3  
**Status:** ACCEPTED

### Decision

Create `src/neuralearn/gradient_check.py` as a reusable module rather than keeping gradient-checking logic only in test files.

### Context

Numerical gradient checking is a validation tool, not a test-specific utility. It should be importable and reusable across test files and potentially in experiment validation.

### Reasoning

- The architecture planned `gradient_check.py` as a separate module
- Reusable across test files and future experiment validation
- Separates the checking logic from specific test cases
- Follows the project's modular structure principle (D05)

### Consequences

- Gradient-checking logic is reusable
- Test files import from the module
- Clear separation of utility vs. test

---

### D26: Epsilon and Tolerance Selection

**Date:** Stage 3  
**Status:** ACCEPTED

### Decision

Use ε=1e-5 for central difference and atol=1e-5, rtol=1e-3 for gradient comparison.

### Context

Central-difference finite differences have O(ε²) truncation error. Too large an ε loses accuracy; too small amplifies floating-point roundoff.

### Reasoning

- ε=1e-5: balances truncation error (~1e-10) against roundoff error (~1e-16 / 1e-5 ≈ 1e-11)
- atol=1e-5: catches absolute gradient errors larger than the finite-difference approximation error
- rtol=1e-3: allows 0.1% relative error, which is appropriate for floating-point comparisons
- These values are standard in autodiff gradient-checking literature

### Consequences

- Consistent gradient checking across all operations
- Meaningful error thresholds that catch real bugs without false positives

---

## Stage 4 Decisions

### D27: Parameter Subclasses Value

**Date:** Stage 4  
**Status:** ACCEPTED

### Decision

Parameter subclasses Value (IS-A relationship). Parameter IS-A Value.

### Context

We need a trainable leaf node that participates in the computational graph. The alternative is a separate class with an adapter pattern.

### Reasoning

- IS-A relationship means Parameter naturally participates in all Value arithmetic
- No adapter code needed; `.data`/`.grad` interface is identical
- Consistent with how PyTorch `nn.Parameter` extends `Tensor`
- Simpler design with less code

### Consequences

- Parameter inherits all Value operations
- Parameter can be used anywhere a Value is expected
- `requires_grad` flag distinguishes trainable from non-trainable values
- `zero_grad()` method resets gradient to 0.0

---

### D28: Module as Lightweight Base Class

**Date:** Stage 4  
**Status:** ACCEPTED

### Decision

Module is a lightweight base class, not an abstract base class. It provides `forward()`, `parameters()`, `zero_grad()` by convention.

### Context

Neural-network layers need a common interface for forward computation, parameter discovery, and gradient reset. An ABC would enforce implementation but adds complexity.

### Reasoning

- Lightweight: no metaclass overhead, no abstractmethod decorators
- Convention-based: subclasses implement methods by name, not by enforced contract
- Pythonic: duck typing over rigid interfaces
- Easy to understand and extend

### Consequences

- Subclasses must implement `forward()`, `parameters()`, `zero_grad()` correctly
- No compile-time enforcement of interface
- Simple and readable base class

---

### D29: Linear Uses Direct Weight/Bias Parameters

**Date:** Stage 4  
**Status:** ACCEPTED

### Decision

Linear layer stores weight and bias Parameters directly, not by wrapping Neuron objects internally.

### Context

Linear is the core layer. The alternative is composing Neuron objects inside Linear.

### Reasoning

- Direct parameter storage is simpler and more efficient
- No indirection through Neuron objects
- Parameters are flat and easy to collect via `parameters()`
- Matches how frameworks implement Linear

### Consequences

- Linear stores weight matrix and bias vector as Parameters
- `parameters()` returns flat list of all weight + bias Parameters
- Total parameter count: `nout * (nin + 1)`
- Neuron is a teaching abstraction, not used internally by Linear

---

### D30: ReLU/Tanh as Module Wrappers

**Date:** Stage 4  
**Status:** ACCEPTED

### Decision

ReLU and Tanh are Module subclasses that wrap existing Value operations (`Value.relu()`, `Value.tanh()`).

### Context

Activation functions need to participate in the Module hierarchy for composition, but their computation is already implemented in Value.

### Reasoning

- No code duplication: delegates to existing Value methods
- Consistent Module interface: all layers are Modules
- Simple implementation: `forward()` just calls the Value method
- Gradient computation is already correct in Value

### Consequences

- ReLU/Tanh are thin wrappers
- Forward pass delegates to Value.relu()/tanh()
- Backward pass uses existing Value backward rules
- Consistent with the Module interface

---

## Stage 5 Decisions

### D31: Loss Functions as Plain Functions

**Date:** Stage 5  
**Status:** ACCEPTED

### Decision

Loss functions are implemented as plain functions (`mse_loss`, `binary_cross_entropy`), not as Module subclasses.

### Context

Loss functions compute a scalar from predictions and targets. Unlike layers, they have no trainable parameters, no state to manage, and no need for `zero_grad()`.

### Reasoning

- No parameters means `parameters()` would return `[]` — Module adds no value
- No state means no `zero_grad()` needed
- Plain functions are simpler, more Pythonic, and easier to understand
- Consistent with functional programming style for pure computations
- Matches the mathematical notation: `L = f(ŷ, y)`

### Consequences

- Loss functions are stateless callables
- No Module inheritance overhead
- Gradient flow works naturally through Value operations inside the function
- Simpler API: `loss = mse_loss(predictions, targets)`

---

### D32: BCE Probability Clipping via ReLU

**Date:** Stage 5  
**Status:** ACCEPTED

### Decision

BCE clips predictions to `[eps, 1-eps]` using existing Value operations:
`clipped = eps + (p - eps).relu() - (p - (1-eps)).relu()`

### Context

BCE requires `log(p)` and `log(1-p)`, which are undefined for `p <= 0` and `p >= 1`. Predictions from a neural network can be any real number.

### Reasoning

- Uses existing `relu()` operation — no new Value methods needed
- Clipping is differentiable everywhere except exact boundary points
- Boundary gradients are 0 (matching ReLU convention at 0)
- For typical predictions away from 0 and 1, gradients flow correctly
- Default `eps=1e-7` provides sufficient numerical stability

### Consequences

- BCE works with any real-valued prediction
- log(0) is prevented by clipping
- Gradient is 0 at exact boundaries (non-issue in practice)
- No need for epsilon-based gradient smoothing

---

### D33: Mean Reduction for Both Losses

**Date:** Stage 5  
**Status:** ACCEPTED

### Decision

Both MSE and BCE use mean reduction (average over samples), not sum reduction.

### Context

Loss functions must reduce multiple per-sample losses to a single scalar. The choice between mean and sum affects the learning rate scale.

### Reasoning

- Mean reduction makes the loss scale-independent of batch size
- Consistent with PyTorch's default (`reduction='mean'`)
- More intuitive: "average error per sample"
- Easier to compare across different batch sizes

### Consequences

- Loss gradient is scaled by `1/n`
- Learning rate is independent of batch size
- Consistent with standard deep-learning practice

---

## Stage 6 Decisions

### D34: Optimizers as Classes

**Date:** Stage 6  
**Status:** ACCEPTED

### Decision

Optimizers are implemented as classes (SGD, MomentumSGD, Adam), not as functions.

### Context

SGD could be a stateless function, but Momentum and Adam maintain per-parameter state (velocity, moments, timestep). A class-based API is consistent across all three.

### Reasoning

- Momentum and Adam require per-parameter state that persists across step() calls
- Class-based API is consistent: all optimizers have step() and zero_grad()
- State is naturally stored as instance attributes
- Matches PyTorch/TensorFlow convention ( familiar API)

### Consequences

- Consistent API: optimizer.step(), optimizer.zero_grad()
- State naturally persists across calls
- Easy to inspect optimizer state for debugging

---

### D35: Optimizers Accept List[Parameter]

**Date:** Stage 6  
**Status:** ACCEPTED

### Decision

Optimizers accept `List[Parameter]` directly, not Module objects.

### Context

Optimizers need to update Parameters. Coupling to Module would create unnecessary dependency.

### Reasoning

- Decouples optimizer from model architecture
- Users extract parameters via model.parameters() and pass to optimizer
- Consistent with PyTorch convention
- Enables mixing parameters from multiple modules

### Consequences

- Optimizer is independent of model structure
- User is responsible for passing correct parameters
- Simple, composable design

---

### D36: Parameter State Keyed by id()

**Date:** Stage 6  
**Status:** ACCEPTED

### Decision

Per-parameter optimizer state is keyed by `id(parameter)`, not by the parameter object itself.

### Context

Python dict keys must be hashable. Value/Parameter may not implement __hash__ consistently, and using object identity is more robust.

### Reasoning

- id() is stable for the lifetime of an object
- Avoids hash/equality issues with Value objects
- State is correctly associated with specific Parameter instances
- Two Parameters with same data but different identity get separate state

### Consequences

- State is correctly isolated per Parameter
- No accidental state sharing between unrelated Parameters
- State is lost if Parameter is garbage collected (acceptable)

---

### D37: optimizer.zero_grad() Delegates to Parameter.zero_grad()

**Date:** Stage 6  
**Status:** ACCEPTED

### Decision

optimizer.zero_grad() calls p.zero_grad() for each parameter, reusing existing infrastructure.

### Context

Parameter.zero_grad() already exists and correctly resets gradients. Duplicating this logic would be wasteful.

### Reasoning

- Single source of truth for gradient reset
- No duplication of logic
- Consistent behavior whether calling parameter.zero_grad() or optimizer.zero_grad()
- Module.zero_grad() also delegates the same way

### Consequences

- gradient reset is consistent everywhere
- No need to maintain separate gradient-clearing logic

---

## Stage 7 Decisions

### D38: Trainer as Class Connecting Model, Loss, Optimizer

**Date:** Stage 7  
**Status:** ACCEPTED

### Decision

Implement a `Trainer` class that accepts `model`, `loss_fn`, and `optimizer` at construction, providing `fit()` and `evaluate()` methods.

### Context

Stage 7 needs to connect the existing components (Module, loss functions, optimizers) into a reusable training API. The alternative is a standalone training function or a more complex framework.

### Reasoning

- Class holds references to model, loss_fn, optimizer — natural grouping
- `fit()` runs the training loop with configurable epochs
- `evaluate()` provides inference-only loss computation
- Matches the conceptual API: trainer orchestrates the training lifecycle
- Simple, readable, and testable

### Consequences

- Training API is clean: `trainer.fit(inputs, targets, epochs=100)`
- Evaluation is separate: `trainer.evaluate(inputs, targets)`
- Trainer does not own model state — model owns parameters, optimizer owns update state

---

### D39: Per-Sample Training (No Batching)

**Date:** Stage 7  
**Status:** ACCEPTED

### Decision

The training loop processes one sample at a time. For each sample: forward → loss → backward → optimizer.step → optimizer.zero_grad.

### Context

The scalar autodiff framework processes individual scalar values. There is no Tensor abstraction or batched matrix operations. Per-sample training is the only approach genuinely supported by the current architecture.

### Reasoning

- Consistent with scalar Value architecture
- No fake vectorization or batching abstraction
- Simple and correct for the current scope
- Dataset/DataLoader batching belongs to Stage 8
- Gradient accumulation across samples can be added later if needed

### Consequences

- Each sample triggers its own backward pass and parameter update
- Gradient reset (zero_grad) happens after each sample
- Training is correct but slower than batched training (acceptable for educational scope)

---

### D40: Training Data as List of (input, target) Pairs

**Date:** Stage 7  
**Status:** ACCEPTED

### Decision

Training data is represented as parallel lists: `inputs` (list of input sequences) and `targets` (list of target values/sequences).

### Context

Stage 7 needs a simple data representation. A proper Dataset/DataLoader abstraction belongs to Stage 8.

### Reasoning

- Parallel lists are the simplest Python data structure
- Compatible with `zip(inputs, targets)` iteration
- No new classes or abstractions needed
- Input format matches existing layer APIs: `model([Value(1.0), Value(2.0)])`
- Target format matches existing loss APIs: `mse_loss(predictions, [target_value])`

### Consequences

- Users provide `inputs = [[1.0, 2.0], [3.0, 4.0]]` and `targets = [5.0, 11.0]`
- No random access, shuffling, or batching — those belong to Stage 8
- Simple API suitable for small educational datasets

---

### D41: Per-Epoch History (Mean Loss)

**Date:** Stage 7  
**Status:** ACCEPTED

### Decision

Training history records the mean loss across all samples for each epoch: `{"loss": [mean_loss_epoch_0, mean_loss_epoch_1, ...]}`.

### Context

Training needs to return useful information for monitoring and debugging. The history structure should be simple, deterministic, and easy to plot later.

### Reasoning

- Mean loss per epoch is the standard training metric
- Simple list of floats — easy to inspect, log, and plot
- No visualization dependency
- Deterministic: same data + same model = same history

### Consequences

- History is a dict with key "loss" mapping to a list of floats
- Loss values are ordinary Python floats (not Values)
- No per-sample loss tracking (too verbose for this stage)

---

### D42: Evaluation Without Parameter Modification

**Date:** Stage 7  
**Status:** ACCEPTED

### Decision

`evaluate()` performs forward passes and computes loss but does NOT call backward(), optimizer.step(), or zero_grad(). Parameters are not modified.

### Context

Evaluation/inference must not alter trainable state. This is a fundamental distinction from training.

### Reasoning

- Evaluation is loss computation only
- No gradient computation needed (no backward)
- No parameter updates (no step)
- No gradient accumulation (no zero_grad)
- Returns same history format for consistency

### Consequences

- Calling evaluate() before/after training gives the same loss for the same model state
- evaluate() is side-effect-free with respect to model parameters
- Users can monitor validation loss during training by calling evaluate() separately

---

## Decision Summary

| ID | Decision | Status |
|----|----------|--------|
| D01 | Python as language | ACCEPTED |
| D02 | NumPy as dependency | ACCEPTED |
| D03 | Exclude DL frameworks | ACCEPTED |
| D04 | Documentation-first | ACCEPTED |
| D05 | Modular structure | ACCEPTED |
| D06 | Separate source/tests/experiments | ACCEPTED |
| D07 | Educational correctness priority | ACCEPTED |
| D08 | Staged development | ACCEPTED |
| D09 | Value class as core | ACCEPTED |
| D10 | Reverse-mode autodiff | ACCEPTED |
| D11 | Topological ordering | ACCEPTED |
| D12 | Layers separate from core | ACCEPTED |
| D13 | Optimizers independent | ACCEPTED |
| D14 | Experiments separate | ACCEPTED |
| D15 | Package name "neuralearn" | ACCEPTED |
| D16 | Internal field naming (_prev, _op, _backward) | ACCEPTED |
| D17 | All ops in single value.py | ACCEPTED |
| D18 | Gradient checking in tests | ACCEPTED |
| D19 | Repeated backward call semantics | ACCEPTED |
| D20 | Division as direct operation | ACCEPTED |
| D21 | Log domain restriction (x > 0) | ACCEPTED |
| D22 | ReLU at zero convention (grad=0) | ACCEPTED |
| D23 | Reciprocal as convenience operation | ACCEPTED |
| D24 | All new ops in value.py | ACCEPTED |
| D25 | Gradient checking as separate module | ACCEPTED |
| D26 | Epsilon=1e-5, atol=1e-5, rtol=1e-3 | ACCEPTED |
| D27 | Parameter subclasses Value (IS-A) | ACCEPTED |
| D28 | Module as lightweight base class | ACCEPTED |
| D29 | Linear uses direct weight/bias | ACCEPTED |
| D30 | ReLU/Tanh as Module wrappers | ACCEPTED |
| D31 | Loss functions as plain functions | ACCEPTED |
| D32 | BCE clips via relu() | ACCEPTED |
| D33 | Mean reduction for both losses | ACCEPTED |
| D34 | Optimizers as classes | ACCEPTED |
| D35 | Optimizers accept List[Parameter] | ACCEPTED |
| D36 | State keyed by id() | ACCEPTED |
| D37 | zero_grad delegates to Parameter | ACCEPTED |
| D38 | Trainer as class connecting model, loss, optimizer | ACCEPTED |
| D39 | Per-sample training (no batching) | ACCEPTED |
| D40 | Training data as list of (input, target) pairs | ACCEPTED |
| D41 | Per-epoch history (mean loss) | ACCEPTED |
| D42 | Evaluation without parameter modification | ACCEPTED |

---

## Future Decisions

The following decisions will be made during implementation:

- Specific backward function implementations
- Gradient checking tolerance values
- Default hyperparameters
- Visualization library choices
- Testing framework configuration
- Documentation format details

---

## Stage 1 Decisions

### D16: Internal Naming Convention for Value Fields

**Date:** Stage 1  
**Status:** ACCEPTED

### Decision

Use `_prev`, `_op`, `_backward` (underscore-prefixed) for internal computational graph fields on `Value`.

### Context

The `Value` class needs fields for parents, operation name, and backward function. These are internal implementation details, not part of the public numerical API.

### Reasoning

- `_prev`, `_op`, `_backward` clearly communicate these are internal
- Public API remains clean: `data`, `grad`, `backward()`
- Follows Python convention for internal attributes
- Prevents accidental external modification of graph structure

### Consequences

- Clean public API
- Internal graph structure is encapsulated
- Consistent naming across implementation

---

### D17: All Operations in Single value.py Module

**Date:** Stage 1  
**Status:** ACCEPTED

### Decision

Implement the `Value` class and all Stage 1 operations (+, *, -, **, neg) in a single `value.py` module.

### Context

For Stage 1, the total code is small enough (~200-300 lines) to fit in one file without sacrificing readability. Splitting into separate operation files would add indirection without benefit at this scale.

### Reasoning

- Keeps related code together
- Easy to read and understand the complete system
- Operations are defined as methods on `Value` (natural grouping)
- Can be split later if the file grows too large

### Consequences

- Single file to read for complete understanding
- No import ceremony between related code
- May need splitting in later stages (acceptable)

---

### D18: Numerical Gradient Checking in Tests

**Date:** Stage 1  
**Status:** ACCEPTED

### Decision

Implement numerical gradient checking (finite differences) directly in the test file rather than as a separate utility module.

### Context

Stage 1 is the first validation of correctness. A focused gradient-checking utility in the test file is sufficient. A separate `gradient_check.py` module (per the architecture) can be created in Stage 3 when the scope demands it.

### Reasoning

- Keeps Stage 1 scope minimal
- Test file is self-contained for Stage 1 validation
- Can be refactored to a utility module later
- Avoids premature abstraction

### Consequences

- Stage 1 tests are self-contained
- Gradient checking is immediately visible in test code
- Refactoring path exists for later stages

---

### D19: Repeated Backward Call Semantics

**Date:** Stage 1  
**Status:** ACCEPTED

### Decision

Define `backward()` semantics for repeated calls: each call computes fresh gradients and **adds** them to leaf-node gradients. Intermediate (non-leaf) node gradients are reset to 0 before each traversal to prevent stale values from corrupting the current pass.

### Context

The initial implementation only reset `self.grad = 1.0` on the root node before traversing. Intermediate nodes retained stale gradients from previous backward calls. When `backward()` was called a second time, the stale intermediate gradients caused incorrect gradient flow — leaf gradients were wrong (e.g., 96.0 instead of the correct 64.0 for `z = (x*x)^2`).

### Reasoning

- **Leaf gradients accumulate:** Useful for gradient accumulation across mini-batches or repeated loss evaluations.
- **Intermediate gradients are fresh:** Each backward pass computes the full chain rule from scratch. Stale intermediate values would corrupt the computation.
- **Root gradient is always 1.0:** `d(output)/d(output) = 1` by definition; it does not accumulate.
- This matches the mathematical model: each `backward()` call adds the gradient `∇f(x)` to the existing leaf gradients.

### Implementation

```python
# Before traversal:
for node in topo_order:
    if node is self:         node.grad = 1.0   # root
    elif len(node._prev) > 0: node.grad = 0.0   # intermediate (non-leaf)
    # else: leaf — untouched (accumulate)
```

### Consequences

- Repeated `backward()` calls on graphs with intermediate nodes produce mathematically correct accumulated leaf gradients.
- Existing tests (leaf-only graphs) continue to pass unchanged.
- No explicit `zero_grad()` method is needed for the core Value class (will be added in later stages for training).

---
