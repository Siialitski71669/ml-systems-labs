# Lab 3 — Architectures and Their Costs

## 1. Goal

Compare MLP, CNN, and attention-based neural network architectures on Fashion-MNIST in terms of classification accuracy, trainable parameters, computational cost, and inference latency.

## 2. Method

- Dataset: Fashion-MNIST.
- Models: MLP, CNN, and Transformer-based Attention model.
- Optimizer: Adam.
- Learning rate: 0.001.
- Training: 5 epochs per model.
- Device: CPU.
- Random seed: 42.
- Inference latency: each model was warmed up for 10 forward passes. Single-image latency was measured 100 times; batch-size-128 latency was measured 5 times. The median was reported.
- Latency measurements used newly initialized models with random weights, so they measure architecture-level inference cost rather than the trained models.

## 3. Classification Results

| Model | Trainable parameters | Test accuracy |
|---|---:|---:|
| MLP | 235,146 | 87.03% |
| CNN | 20,490 | 87.74% |
| Attention | 70,794 | 84.04% |

## 4. Computational Cost

| Model | MACs | Approximate FLOPs |
|---|---:|---:|
| MLP | 234,752 | 469,504 |
| CNN | 1,031,744 | 2,063,488 |
| Attention | 591,488 | 1,182,976 |

MACs were estimated with THOP. Approximate FLOPs were calculated as twice the MAC count. Actual totals can differ depending on how operations such as bias additions, activations, pooling, and normalization are counted.

### Representative hand calculations

For a fully connected layer:

Parameters = input_features × output_features + output_features (biases).

For the MLP:

- Layer 1: 784 × 256 + 256 = 200,960.
- Layer 2: 256 × 128 + 128 = 32,896.
- Output layer: 128 × 10 + 10 = 1,290.
- Total: 235,146 parameters.

For a convolutional layer:

Parameters = (kernel_height × kernel_width × input_channels + 1) × output_channels.

- Conv1: (3 × 3 × 1 + 1) × 16 = 160.
- Conv2: (3 × 3 × 16 + 1) × 32 = 4,640.
- Fully connected layer: 7 × 7 × 32 × 10 + 10 = 15,690.
- Total CNN parameters: 20,490.

Manual parameter counts match the model parameter counts reported by the profiling output. FLOPs are estimates and depend on the operation-counting convention.

## 5. Inference Latency

Median latency measured on CPU:

| Model | Batch size 1 (100 repeats) | Batch size 128 (5 repeats) |
|---|---:|---:|
| MLP | 0.2442 ms | 0.7366 ms |
| CNN | 0.6036 ms | 12.0378 ms |
| Attention | 1.6352 ms | 7.6804 ms |

These latency measurements used randomly initialized models. They should be interpreted as measurements of the architectures on this machine, not as a direct comparison of trained-model performance. Results can vary with hardware, software versions, and system load.

## 6. Conclusions

1. CNN achieved the highest test accuracy in this experiment at 87.74%, slightly outperforming MLP at 87.03%.
2. CNN had the fewest trainable parameters (20,490), while MLP had the most (235,146). A smaller parameter count does not necessarily imply fewer operations or lower latency.
3. Model selection should consider accuracy, parameter count, computational cost, and measured latency together. Reliable comparisons require consistent hardware and measurement procedures.
