# Intro to Neural Network (Micrograd Walkthrough)

This repository contains a single notebook (`main.ipynb`) that demonstrates how neural network training works under the hood by re-creating core ideas from Andrej Karpathy’s **micrograd**: a tiny autograd engine and neural network library.

The notebook builds up from basic calculus intuition to a minimal automatic differentiation engine and a small multi-layer perceptron (MLP) trained with gradient descent.

---

## Contents Overview

### 1) Function, Derivative, and Numerical Gradients

- Defines a simple quadratic function and plots its shape.
- Demonstrates numerical differentiation using finite differences.
- Explains how small input changes affect outputs.

### 2) Scalar Computational Graphs

- Constructs a scalar expression with multiple inputs.
- Tracks how changes in inputs affect outputs (slopes).
- Manually computes gradients using the chain rule.

### 3) Building a Minimal Autograd Engine (`Value`)

- Implements a `Value` class supporting:
  - `+`, `*`, `**`, `-`, `/`, `tanh`, `exp`
  - gradient accumulation
  - automatic backpropagation via a topological sort
- Visualizes graphs using `graphviz`.

### 4) Neuron Example (Forward + Backward)

- Builds a single neuron with tanh activation.
- Runs both manual and automatic backpropagation.
- Verifies gradients and shows the computation graph.

### 5) PyTorch Comparison

- Repeats the neuron example using PyTorch autograd.
- Compares gradients to the custom implementation.

### 6) MLP from Scratch

- Defines:
  - `Neuron`
  - `Layer`
  - `MLP`
- Runs forward passes on a tiny dataset.
- Computes mean squared error (MSE) loss.
- Performs gradient descent training loop.
- Shows loss decreasing and final predictions.

---

## Dependencies

The notebook uses:

- `numpy`
- `matplotlib`
- `torch`
- `graphviz` (both Python package and system install)

Install Python packages:

```bash
pip install numpy matplotlib torch graphviz
```

Install Graphviz (system):

- macOS (Homebrew):
  ```bash
  brew install graphviz
  ```

> If you use VS Code notebooks, you may need both the `graphviz` pip package and the system `graphviz` install.

---

## How to Run

1. Open `main.ipynb` in Jupyter or VS Code.
2. Run cells top to bottom in order.
3. Optional: enable inline plotting (`%matplotlib inline` is already included).

---

## Notes

- The notebook is educational and focuses on clarity rather than performance.
- All gradients are computed for scalar values to make the mechanics of backpropagation explicit.

---

## Credits

Inspired by Andrej Karpathy’s micrograd project:

- https://github.com/karpathy/micrograd
