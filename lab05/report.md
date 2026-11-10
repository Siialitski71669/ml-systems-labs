# Lab 05 — Frameworks and Automatic Differentiation

## 1. Objectives

The objectives of this laboratory were:
- Implement a scalar automatic differentiation engine.
- Construct computation graphs during forward operations.
- Implement reverse-mode automatic differentiation.
- Validate gradients against PyTorch and numerical estimates.
- Train a small neural network on the make_moons dataset.
- Compare custom engine performance with PyTorch.
- Inspect a PyTorch computation graph.
- Compare eager and traced execution.
- Investigate contiguous and non-contiguous tensor layouts.

## 2. Automatic Differentiation Engine

A custom `Value` class was implemented in `src/engine.py`.

Each node stores:
- Scalar data.
- Gradient.
- Parent nodes.
- Operation information.
- A local backward function.

Implemented operations:
- Addition.
- Multiplication.
- Power.
- ReLU.
- Negation.
- Subtraction.
- Division.
- Exponential.
- Logarithm.
- Sigmoid.

The backward pass uses topological sorting and reverse traversal. Gradients are accumulated using addition, allowing multiple computational paths to contribute to the same node.

### Unit tests

All basic tests passed:
- Main expression.
- Gradient accumulation.
- ReLU.

Gradient accumulation was verified with the expression `x * x + x` at `x = 3`, producing a gradient of `7`.

## 3. Gradient Verification

The custom engine was compared with PyTorch autograd and central finite differences.

Test expression:

`f(a,b,c) = ReLU((a*b+c)^2)`

Input values:
- a = 2
- b = 3
- c = 4

Output: `100.0`

| Method | da | db | dc |
|---|---:|---:|---:|
| Custom engine | 60.0 | 40.0 | 20.0 |
| PyTorch | 60.0 | 40.0 | 20.0 |
| Numerical | 59.99999999 | 40.00000001 | 20.00000002 |

Maximum absolute differences:
- Custom vs PyTorch: `0.0`
- Custom vs numerical: `0.000000020559`

Both differences are below the required `1e-6` threshold.

## 4. Neural Network Training

A small neural network was trained on the `make_moons` dataset.

Configuration:
- Dataset: 120 samples.
- Architecture: 2 → 8 → 1.
- Hidden activation: ReLU.
- Output activation: Sigmoid.
- Loss: Binary Cross-Entropy.
- Epochs: 20.
- Learning rate: 0.05.
- Performance measurements: median of 5 runs.

Results:

| Metric | Custom engine | PyTorch |
|---|---:|---:|
| Accuracy | 87.5% | 87.5% |
| Median training time | 1.2791 s | 1.7619 s |

The custom engine was faster in this particular experiment. This comparison uses per-sample Python training loops and should not be interpreted as a general performance advantage over PyTorch.

## 5. PyTorch Computation Graph

A model with the architecture below was inspected using `torch.jit.trace` and its inlined graph:

`Linear(16, 32) → ReLU → Linear(32, 1) → Sigmoid`

The graph exposed the corresponding linear, ReLU, and sigmoid operations, together with tensor shapes and model parameters.

## 6. Eager vs Traced Execution

The eager model and traced model produced identical outputs.

Maximum absolute output difference: `0.0`.

Latest measured results:

| Execution mode | Median time per call |
|---|---:|
| Eager | 0.0786 ms |
| Traced | 0.0527 ms |

The traced model was approximately 1.49 times faster in this measurement.

## 7. Tensor Layout Analysis

Two tensors were inspected:

| Property | Contiguous | Transposed |
|---|---|---|
| Shape | 1024 × 1024 | 1024 × 1024 |
| Dtype | float32 | float32 |
| Contiguous | True | False |
| Stride | (1024, 1) | (1, 1024) |

Measured operation: `(tensor * 2 + 1).sum()`.

| Layout | Median time |
|---|---:|
| Contiguous | 0.2646 ms |
| Non-contiguous | 0.2434 ms |

The non-contiguous tensor was slightly faster in this particular measurement. This demonstrates that memory layout does not guarantee a fixed performance difference for every operation and benchmark configuration.

## 8. Conclusions

The laboratory successfully demonstrated the implementation of a scalar automatic differentiation engine and reverse-mode backpropagation.

The custom gradients matched PyTorch exactly and numerical gradients within the required tolerance.

The custom engine successfully trained a small neural network. The PyTorch experiment demonstrated computation graph inspection and the performance difference between eager and traced execution.

Tensor inspection confirmed that transposition changes strides and contiguity. The measured timing difference was small and depended on the tested operation.

All implemented tests and framework checks passed.
