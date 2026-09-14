# 05 — Research Plan

**Project:** Neural Network From Scratch  
**Version:** 1.0  
**Status:** Stage 0 — Planning

## Overview

This project is treated as a miniature research-engineering effort. This document defines the theoretical areas that must be studied while building the framework.

**Research Philosophy:** Understand → Implement → Verify

## Theoretical Areas

### T1: Derivatives

**What to Understand:**
- Definition of derivative as rate of change
- Geometric interpretation (slope of tangent line)
- Derivative notation (Leibniz, Lagrange, Newton)
- Derivatives of common functions (polynomial, exponential, logarithmic, trigonometric)

**Why It Matters:**
- Derivatives are the foundation of gradient-based optimization
- Understanding derivatives enables implementing backward functions
- Derivative properties guide numerical stability

**Connection to Implementation:**
- Every backward function computes a derivative
- Chain rule combines derivatives through composition
- Numerical gradient checking approximates derivatives

**Evidence of Understanding:**
- Can compute derivatives of all operations in the framework
- Can explain geometric meaning of gradients
- Can derive backward functions from forward operations

### T2: Partial Derivatives

**What to Understand:**
- Definition of partial derivative
- Gradient as vector of partial derivatives
- Partial derivatives in multivariate functions
- Notation and computation

**Why It Matters:**
- Neural networks are functions of many variables
- Gradients are vectors of partial derivatives
- Each parameter has its own partial derivative

**Connection to Implementation:**
- Parameters are multivariate
- Gradients are computed with respect to each parameter
- Backward functions compute partial derivatives

**Evidence of Understanding:**
- Can compute partial derivatives of multivariate functions
- Can explain gradient vector
- Can implement partial derivative computation

### T3: Chain Rule

**What to Understand:**
- Single-variable chain rule
- Multivariate chain rule
- Chain rule for compositions of functions
- Implicit differentiation via chain rule

**Why It Matters:**
- Chain rule is the mathematical foundation of backpropagation
- Backpropagation is reverse-mode automatic differentiation via chain rule
- Every gradient computation uses chain rule

**Connection to Implementation:**
- Backward functions implement chain rule
- Gradient propagation applies chain rule sequentially
- Topological ordering ensures correct chain rule application

**Evidence of Understanding:**
- Can apply chain rule to complex compositions
- Can derive backpropagation from chain rule
- Can implement chain rule in backward functions

### T4: Computational Graphs

**What to Understand:**
- Directed acyclic graphs (DAGs)
- Nodes and edges in computational graphs
- Topological ordering
- Forward and backward traversal

**Why It Matters:**
- Computational graphs represent the sequence of operations
- Graph structure determines gradient computation order
- Graph enables automatic differentiation

**Connection to Implementation:**
- Values are nodes in the computational graph
- Operations create edges between nodes
- Topological sort enables backward pass

**Evidence of Understanding:**
- Can draw computational graphs for expressions
- Can perform topological sorting
- Can explain how graphs enable autodiff

### T5: Gradients

**What to Understand:**
- Gradient definition and properties
- Direction of steepest ascent
- Gradient as vector of partial derivatives
- Gradient of scalar functions

**Why It Matters:**
- Gradients indicate how to change parameters to decrease loss
- Gradient descent uses gradients for optimization
- Gradient computation is the core of training

**Connection to Implementation:**
- Backward pass computes gradients
- Optimizers use gradients to update parameters
- Gradient checking validates gradient computation

**Evidence of Understanding:**
- Can compute gradients analytically
- Can explain gradient descent
- Can verify gradients numerically

### T6: Reverse-Mode Automatic Differentiation

**What to Understand:**
- Forward-mode vs reverse-mode autodiff
- Why reverse-mode is efficient for scalar outputs
- Tape-based vs graph-based autodiff
- Memory-computation trade-offs

**Why It Matters:**
- Reverse-mode autodiff is how backpropagation works
- This is the core technical contribution of the project
- Understanding autodiff enables debugging and extension

**Connection to Implementation:**
- The Value class is the "tape" recording operations
- Backward pass is reverse-mode autodiff
- Topological ordering enables correct computation

**Evidence of Understanding:**
- Can implement reverse-mode autodiff
- Can explain why reverse-mode is efficient for neural networks
- Can compare forward-mode and reverse-mode

### T7: Backpropagation

**What to Understand:**
- Backpropagation as reverse-mode autodiff
- Forward pass and backward pass
- Gradient accumulation
- Chain rule application in backpropagation

**Why It Matters:**
- Backpropagation is the algorithm for training neural networks
- Understanding backpropagation is understanding deep learning
- This is the central algorithm the project implements

**Connection to Implementation:**
- Forward pass builds computational graph
- Backward pass propagates gradients
- Gradient accumulation handles repeated values

**Evidence of Understanding:**
- Can implement backpropagation from scratch
- Can explain each step of backpropagation
- Can debug training failures using backpropagation understanding

### T8: Parameter Optimization

**What to Understand:**
- Gradient descent algorithm
- Learning rate and its effects
- Local minima and saddle points
- Convergence properties

**Why It Matters:**
- Optimization is how models learn
- Learning rate is the most important hyperparameter
- Understanding optimization enables debugging training

**Connection to Implementation:**
- Optimizers implement optimization algorithms
- Learning rate is a key optimizer parameter
- Training loop orchestrates optimization

**Evidence of Understanding:**
- Can explain gradient descent convergence
- Can diagnose learning rate issues
- Can implement optimization algorithms

### T9: Gradient Descent

**What to Understand:**
- Batch, stochastic, and mini-batch gradient descent
- Learning rate schedules
- Convergence criteria
- Step size and convergence

**Why It Matters:**
- Gradient descent is the foundation of all optimizers
- Understanding gradient descent enables understanding advanced optimizers
- Learning rate determines training stability

**Connection to Implementation:**
- SGD implements basic gradient descent
- Momentum and Adam extend gradient descent
- Training loop manages gradient descent iterations

**Evidence of Understanding:**
- Can implement gradient descent
- Can explain learning rate effects
- Can diagnose convergence issues

### T10: Learning Rates

**What to Understand:**
- Learning rate as step size
- Too high: divergence, oscillation
- Too low: slow convergence, local minima
- Adaptive learning rates (Adam, RMSprop)

**Why It Matters:**
- Learning rate is the most critical hyperparameter
- Wrong learning rate causes training failure
- Adaptive methods automate learning rate selection

**Connection to Implementation:**
- SGD uses fixed learning rate
- Adam computes adaptive learning rates
- Experiments study learning rate effects

**Evidence of Understanding:**
- Can explain learning rate effects
- Can diagnose learning rate problems
- Can implement adaptive learning rate methods

### T11: Activation Functions

**What to Understand:**
- Why nonlinearity is necessary
- ReLU: definition, properties, advantages
- Tanh: definition, properties, advantages
- Dead neurons, vanishing gradients

**Why It Matters:**
- Activations enable nonlinear function approximation
- Choice of activation affects training dynamics
- Understanding activations enables debugging

**Connection to Implementation:**
- ReLU and Tanh are implemented as layers
- Backward functions compute activation derivatives
- Experiments compare activation effects

**Evidence of Understanding:**
- Can explain why nonlinearity is necessary
- Can implement activation backward functions
- Can diagnose activation-related training issues

### T12: Loss Functions

**What to Understand:**
- Loss as measure of prediction error
- MSE for regression
- Binary Cross-Entropy for classification
- Loss surface and optimization

**Why It Matters:**
- Loss defines what the model learns
- Different losses suit different problems
- Loss surface affects optimization difficulty

**Connection to Implementation:**
- Loss functions compute scalar loss
- Backward functions propagate loss gradients
- Experiments use different losses

**Evidence of Understanding:**
- Can choose appropriate loss for problem
- Can implement loss backward functions
- Can explain loss surface properties

### T13: Overfitting

**What to Understand:**
- Definition of overfitting
- Training vs test performance
- Model complexity and overfitting
- Regularization techniques

**Why It Matters:**
- Overfitting is a fundamental challenge in ML
- Understanding overfitting enables better model design
- Regularization is essential for generalization

**Connection to Implementation:**
- Experiments observe overfitting behavior
- Future work may add regularization
- Understanding overfitting guides architecture choices

**Evidence of Understanding:**
- Can identify overfitting in training curves
- Can explain causes of overfitting
- Can propose regularization strategies

### T14: Generalization

**What to Understand:**
- Definition of generalization
- Training vs test performance gap
- Factors affecting generalization
- Bias-variance trade-off

**Why It Matters:**
- Generalization is the goal of ML
- Understanding generalization enables better models
- Evaluating generalization is essential

**Connection to Implementation:**
- Experiments measure generalization (train vs test)
- Model architecture affects generalization
- Training duration affects generalization

**Evidence of Understanding:**
- Can measure generalization
- Can explain bias-variance trade-off
- Can propose strategies to improve generalization

### T15: Numerical Gradient Checking

**What to Understand:**
- Finite-difference approximation
- Forward difference, central difference
- Error sources (truncation, roundoff)
- Appropriate tolerance selection

**Why It Matters:**
- Gradient checking verifies autodiff correctness
- Essential for validating implementation
- Catches subtle bugs in backward functions

**Connection to Implementation:**
- Gradient checking is a critical validation tool
- Every gradient must pass gradient checking
- Tolerance selection affects test reliability

**Evidence of Understanding:**
- Can implement gradient checking
- Can explain error sources
- Can select appropriate tolerance

## Research Methodology

### For Each Topic

1. **Read** authoritative sources (textbooks, papers, documentation)
2. **Work through** examples by hand
3. **Implement** in code
4. **Verify** correctness through testing
5. **Document** understanding in 13_LEARNINGS.md

### Recommended Resources

| Topic | Resource Type |
|-------|---------------|
| Derivatives, Chain Rule | Calculus textbook |
| Computational Graphs | CS231n, Deep Learning textbook |
| Backpropagation | CS231n, Neural Networks and Deep Learning |
| Optimization | Deep Learning textbook, optimization course |
| Gradient Checking | CS231n assignments |
| Activation Functions | Deep Learning textbook |
| Loss Functions | Deep Learning textbook |
| Overfitting, Generalization | Machine Learning textbook |

## Research Log

Research findings should be documented in [13_LEARNINGS.md](13_LEARNINGS.md).

Each entry should include:
- Concept
- Initial understanding
- What was implemented
- What was observed
- What was initially wrong
- What was learned
- Mathematical insight
- Engineering insight
- Open questions

## Progress Tracking

Research progress should be tracked alongside implementation progress in [11_PROGRESS_LOG.md](11_PROGRESS_LOG.md).

## Open Questions

Research may uncover open questions. These should be documented and tracked.
