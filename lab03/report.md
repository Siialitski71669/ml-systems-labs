# Lab 3 — Architectures and Their Costs

## 1. Goal

Compare MLP, CNN, and attention-based neural network architectures on Fashion-MNIST, considering classification accuracy, trainable parameters, computational cost, and inference latency.

## 2. Method

- Dataset: Fashion-MNIST.
- Models: MLP, CNN, and Transformer-based attention model.
- Optimizer: Adam.
- Learning rate: 0.001.
- Training duration in the existing experiment: 5 epochs.
- Device: CPU.
- The existing measurements are retained as historical experiment results. They must be regenerated with the updated profiling script before being treated as fully reproducible benchmark results.

## 3. Results

Existing experiment results:

| Model | Trainable parameters | Test accuracy |
|---|---:|---:|
| MLP | 235,146 | 87.14% |
| CNN | 20,490 | 90.05% |
| Attention | 70,794 | 84.01% |

## 4. Computational Cost

| Model | MACs | Approximate FLOPs |
|---|---:|---:|
| MLP | 234,752 | 469,504 |
| CNN | 1,031,744 | 2,063,488 |
| Attention | 591,488 | 1,182,976 |

FLOPs are approximate. Attention implementations may contain operations that are not fully captured by simple MAC-counting tools.

## 5. Inference Latency

The CSV and plot in `results/` contain measurements from the earlier experiment. Latency should be remeasured after warming up each model, repeating measurements at least five times, and reporting the median. Single-image inference should be repeated 100 times and reported separately.

## 6. Conclusions

1. In the existing experiment, CNN achieved the highest test accuracy at 90.05%.
2. CNN had the fewest trainable parameters, showing that a smaller model can outperform a larger one on this dataset.
3. Model selection should consider accuracy, parameter count, computational cost, and measured latency together. Benchmark comparisons should use the same hardware and measurement procedure.
