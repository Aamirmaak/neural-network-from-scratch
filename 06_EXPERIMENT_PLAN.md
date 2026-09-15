# 06 — Experiment Plan

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 9 COMPLETE

## Overview

This document defines the experimental framework for the project. Experiments demonstrate that the implemented system works correctly and enable investigation of training behavior.

**Current State:** Experiments A and B completed. See 12_EXPERIMENT_LOG.md for results.

## Experiment Methodology

### Principles

1. **Reproducibility:** Every experiment must be reproducible from documented configuration
2. **Controlled comparison:** When comparing factors, only one factor varies at a time
3. **Hypothesis-driven:** Each experiment tests a specific hypothesis
4. **Quantitative measurement:** Results are measured, not claimed
5. **Honest reporting:** Failed experiments are documented alongside successes

### Record Structure

Each experiment record must contain:

| Field | Description |
|-------|-------------|
| Experiment ID | Unique identifier |
| Date | When experiment was conducted |
| Objective | What the experiment aims to demonstrate |
| Hypothesis | What we expect to observe |
| Baseline | Reference configuration for comparison |
| Configuration | Full parameter settings |
| Environment | Hardware, software, seeds |
| Metrics | What was measured |
| Results | Actual measurements |
| Interpretation | What the results mean |
| Failure Cases | What went wrong |
| Conclusion | Final assessment |
| Reproducibility Notes | How to reproduce |

### Required Metrics

| Metric | Description | When Used |
|--------|-------------|-----------|
| Training loss | Loss on training data | All experiments |
| Test/validation loss | Loss on held-out data | When applicable |
| Accuracy | Classification accuracy | Classification experiments |
| Convergence speed | Epochs to reach threshold | Comparison experiments |
| Training time | Wall-clock time | Performance comparison |
| Parameter count | Number of trainable parameters | Architecture experiments |
| Memory usage | Peak memory | Optional |

---

## Experiment A — XOR Classification

### Objective

Demonstrate that the implemented system can learn a nonlinear decision boundary.

### Hypothesis

A neural network with at least one hidden layer can learn the XOR function with near-perfect accuracy.

### Dataset

| Property | Value |
|----------|-------|
| Inputs | 2D binary vectors: [0,0], [0,1], [1,0], [1,1] |
| Targets | [0, 1, 1, 0] |
| Training samples | 4 |
| Test samples | 4 (same as training) |

### Model Variables

| Variable | Default | Options |
|----------|---------|---------|
| Hidden size | 8 | [4, 8, 16, 32] |
| Activation | Tanh | [Tanh, ReLU] |
| Output activation | Sigmoid | [Sigmoid] |
| Loss | Binary Cross-Entropy | [BCE] |

### Training Variables

| Variable | Default | Options |
|----------|---------|---------|
| Optimizer | Adam | [SGD, Momentum, Adam] |
| Learning rate | 0.01 | [0.001, 0.01, 0.1, 1.0] |
| Epochs | 1000 | [100, 500, 1000, 5000] |
| Random seed | 42 | Any integer |

### Metrics

| Metric | Description |
|--------|-------------|
| Final training loss | Loss after training |
| Final accuracy | Classification accuracy |
| Convergence epoch | Epoch when loss < 0.01 |
| Decision boundary | Visualization |

### Baseline

Single-layer linear model (should fail to learn XOR).

### Expected Observations

1. Linear model cannot learn XOR (high loss, ~50% accuracy)
2. Hidden layer enables learning (low loss, 100% accuracy)
3. Training loss decreases over epochs
4. Decision boundary is nonlinear

### Controlled Variables

When comparing factors, all other variables are held constant.

---

## Experiment B — Nonlinear Regression

### Objective

Demonstrate that the system can fit complex nonlinear functions.

### Hypothesis

A neural network with sufficient capacity can approximate a nonlinear function with low MSE.

### Dataset

| Property | Value |
|----------|-------|
| Input | 1D: x ∈ [-3, 3] |
| Target | y = sin(x) + 0.1x² |
| Training samples | 100 |
| Test samples | 50 |

### Model Variables

| Variable | Default | Options |
|----------|---------|---------|
| Hidden sizes | [16, 16] | [[8], [16, 16], [32, 32, 32]] |
| Activation | Tanh | [Tanh, ReLU] |
| Output activation | None (linear) | [None] |
| Loss | MSE | [MSE] |

### Training Variables

| Variable | Default | Options |
|----------|---------|---------|
| Optimizer | Adam | [SGD, Momentum, Adam] |
| Learning rate | 0.01 | [0.001, 0.01, 0.1] |
| Epochs | 2000 | [500, 1000, 2000, 5000] |
| Random seed | 42 | Any integer |

### Metrics

| Metric | Description |
|--------|-------------|
| Final training MSE | Mean squared error on training |
| Final test MSE | Mean squared error on test |
| Prediction curve | Visualization |
| Learning curve | Loss vs epoch |

### Baseline

Linear regression (should underfit).

### Expected Observations

1. Linear model underfits (high MSE)
2. Neural network fits well (low MSE)
3. Training loss decreases over epochs
4. Predictions match target function

---

## Controlled Experiments

### Experiment C1 — Optimizer Comparison

**Objective:** Compare SGD, Momentum, and Adam optimizers.

**Hypothesis:** Adam converges faster than Momentum, which converges faster than SGD.

**Controlled Variables:**
- Model: Same architecture for all optimizers
- Learning rate: Same for all (or tuned per optimizer)
- Dataset: Same for all
- Random seed: Same for all

**Independent Variable:** Optimizer type

**Dependent Variables:**
- Convergence speed (epochs to threshold)
- Final loss
- Training time

**Configuration:**

| Optimizer | Learning Rate | Other Parameters |
|-----------|---------------|------------------|
| SGD | 0.01 | None |
| Momentum | 0.01 | momentum=0.9 |
| Adam | 0.01 | beta1=0.9, beta2=0.999 |

**Metrics:**
- Loss vs epoch curves
- Time to convergence
- Final loss value

### Experiment C2 — Learning Rate Study

**Objective:** Investigate effect of learning rate on training.

**Hypothesis:** Optimal learning rate exists; too high causes divergence, too low causes slow convergence.

**Controlled Variables:**
- Model: Same architecture
- Optimizer: Same (Adam)
- Dataset: Same
- Random seed: Same

**Independent Variable:** Learning rate

**Dependent Variables:**
- Convergence speed
- Final loss
- Training stability

**Configuration:**

| Learning Rate | Expected Behavior |
|---------------|-------------------|
| 0.001 | Slow but stable convergence |
| 0.01 | Good convergence |
| 0.1 | Fast but possibly unstable |
| 1.0 | Likely divergence |

**Metrics:**
- Loss vs epoch curves
- Final loss
- Training stability (variance)

### Experiment C3 — Hidden Layer Size

**Objective:** Investigate effect of model capacity on learning.

**Hypothesis:** Larger models learn faster but may overfit; smaller models may underfit.

**Controlled Variables:**
- Architecture type: Same (MLP)
- Activation: Same
- Optimizer: Same
- Learning rate: Same
- Dataset: Same
- Random seed: Same

**Independent Variable:** Hidden layer size

**Dependent Variables:**
- Training loss
- Test loss (if applicable)
- Convergence speed
- Parameter count

**Configuration:**

| Hidden Size | Parameters |
|-------------|------------|
| 4 | Few |
| 8 | Moderate |
| 16 | Many |
| 32 | Very many |

**Metrics:**
- Training and test loss curves
- Parameter count
- Convergence speed

### Experiment C4 — Activation Function Comparison

**Objective:** Compare ReLU and Tanh activations.

**Hypothesis:** Tanh may converge faster for small networks; ReLU may be better for deep networks.

**Controlled Variables:**
- Model architecture: Same
- Optimizer: Same
- Learning rate: Same
- Dataset: Same
- Random seed: Same

**Independent Variable:** Activation function

**Dependent Variables:**
- Convergence speed
- Final loss
- Gradient flow

**Configuration:**

| Activation | Properties |
|------------|------------|
| Tanh | Bounded, smooth |
| ReLU | Unbounded, non-smooth |

**Metrics:**
- Loss vs epoch curves
- Final loss
- Gradient magnitudes

### Experiment C5 — Number of Layers

**Objective:** Investigate effect of network depth.

**Hypothesis:** Deeper networks can learn more complex functions but may be harder to train.

**Controlled Variables:**
- Hidden size per layer: Same
- Activation: Same
- Optimizer: Same
- Learning rate: Same
- Dataset: Same
- Random seed: Same

**Independent Variable:** Number of hidden layers

**Dependent Variables:**
- Training loss
- Test loss (if applicable)
- Convergence speed
- Gradient flow

**Configuration:**

| Layers | Description |
|--------|-------------|
| 1 | Shallow network |
| 2 | Moderate depth |
| 3 | Deeper network |

**Metrics:**
- Training and test loss curves
- Convergence speed
- Gradient magnitudes

---

## Experiment Execution Plan

### Phase 1: Basic Validation (Stages 8-9) — COMPLETE

1. XOR experiment with default configuration ✓
2. Regression experiment with default configuration ✓
3. Verify basic training works ✓

### Phase 2: Controlled Comparisons (Stage 10)

1. Optimizer comparison
2. Learning rate study
3. Hidden layer size study
4. Activation function comparison
5. Number of layers study

### Phase 3: Analysis and Documentation (Stage 11)

1. Analyze all results
2. Create comparison visualizations
3. Document findings
4. Identify failure cases

---

## Result Recording

### Template

```markdown
## Experiment [ID]: [Name]

**Date:** [Date]
**Status:** [COMPLETE/IN PROGRESS]

### Configuration
[Full configuration details]

### Results
[Actual measurements]

### Visualization
[Plots and images]

### Interpretation
[What the results mean]

### Failure Cases
[What went wrong]

### Conclusion
[Final assessment]
```

### Anti-Fabrication Policy

**NEVER FABRICATE RESULTS.**

- Only record actual measurements
- If an experiment fails, document the failure
- If results are unexpected, investigate and document
- Honest failure is more valuable than fabricated success

---

## Reproducibility

### Requirements

Every experiment must be reproducible by:

1. Reading this document
2. Using the documented configuration
3. Setting the documented random seed
4. Running the experiment code
5. Obtaining the same results

### Storage

Results should be stored in:
- `experiments/results/` for result files
- `experiments/figures/` for visualizations
- `12_EXPERIMENT_LOG.md` for experiment records

---

## Open Questions

- Should we use learning rate scheduling?
- Should we implement early stopping?
- Should we add regularization experiments?
- Should we study batch size effects?

These are future considerations, not current requirements.
