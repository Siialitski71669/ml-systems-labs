# Lab 06 — CNN Training Pipeline and Performance Analysis

## Goal

The goal of this laboratory was to build and evaluate a reproducible CNN training pipeline for FashionMNIST, including checkpointing and resuming, gradient accumulation, optimizer and scheduler comparisons, data-loading benchmarks, batch-size benchmarks, and profiling.

## Method

- Dataset: FashionMNIST.
- Model: CNN with 20,490 trainable parameters.
- Reproducibility: seed 42 and deterministic execution.
- Framework: PyTorch, CPU.
- Training configuration: YAML files.
- Checkpointing: model checkpoints saved after training.
- Resume: checkpoint loading tested successfully.
- Gradient accumulation: batch size 64, accumulation steps 4.
- Optimizer comparison: Adam and SGD.
- Schedulers: StepLR, CosineAnnealingLR, ReduceLROnPlateau.
- Batch-size benchmark: 32, 64, 128, 256, 512.
- Data-loader benchmark: num_workers 0, 1, 2, 4, 8.
- Profiling: profiler trace saved in results/profile_trace.json.

## Results

### 1. Optimizer and Scheduler Comparison

All six combinations were trained for 10 epochs with batch size 128.

| Optimizer | Scheduler | Epochs | Batch size | Validation accuracy (%) | Test accuracy (%) | Training time (s) |
|---|---|---:|---:|---:|---:|---:|
| Adam | StepLR | 10 | 128 | 89.68 | 89.39 | 243.41 |
| Adam | CosineAnnealingLR | 10 | 128 | 89.76 | 89.49 | 238.84 |
| Adam | ReduceLROnPlateau | 10 | 128 | 89.94 | 89.08 | 249.75 |
| SGD | StepLR | 10 | 128 | 82.32 | 82.33 | 173.21 |
| SGD | CosineAnnealingLR | 10 | 128 | 82.58 | 82.63 | 135.74 |
| SGD | ReduceLROnPlateau | 10 | 128 | 84.62 | 84.52 | 136.29 |

Adam with CosineAnnealingLR achieved the highest test accuracy of 89.49%. Adam with ReduceLROnPlateau achieved the highest validation accuracy of 89.94%.

### 2. Batch Size Benchmark

Each batch size was tested for 5 epochs.

| Batch size | Epochs | Training time (s) | Throughput (samples/s) | Peak memory (MB) | Validation accuracy (%) |
|---:|---:|---:|---:|---:|---:|
| 32 | 5 | 131.35 | 2093.64 | 356.39 | 90.04 |
| 64 | 5 | 135.30 | 2032.59 | 421.23 | 89.74 |
| 128 | 5 | 281.98 | 975.23 | 422.45 | 88.34 |
| 256 | 5 | 126.88 | 2167.43 | 433.72 | 87.80 |
| 512 | 5 | 123.55 | 2225.80 | 408.62 | 86.24 |

Batch size 512 achieved the highest throughput and shortest training time. Batch size 32 achieved the highest validation accuracy.

### 3. Data Loader Benchmark

Each configuration processed 55,000 samples.

| num_workers | Samples | Total time (s) | Throughput (samples/s) | Data loading (%) | Forward (%) | Backward (%) | Optimizer (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 55000 | 15.57 | 3532.79 | 47.53 | 23.16 | 25.82 | 3.49 |
| 1 | 55000 | 8.72 | 6305.09 | 29.00 | 32.16 | 33.30 | 5.54 |
| 2 | 55000 | 6.88 | 7997.88 | 5.00 | 44.47 | 43.70 | 6.83 |
| 4 | 55000 | 7.17 | 7670.06 | 4.82 | 45.52 | 42.91 | 6.75 |
| 8 | 55000 | 6.94 | 7928.34 | 5.00 | 44.78 | 43.39 | 6.83 |

The best throughput was achieved with num_workers = 2, reaching 7997.88 samples per second. Increasing the number of workers beyond 2 did not improve performance.

### 4. Gradient Accumulation

The gradient accumulation experiment used batch size 64, 4 accumulation steps, and 5 epochs.

| Epoch | Training loss | Validation loss | Validation accuracy (%) |
|---:|---:|---:|---:|
| 1 | 0.7354 | 0.4725 | 83.40 |
| 2 | 0.4257 | 0.4060 | 85.68 |
| 3 | 0.3724 | 0.3947 | 85.72 |
| 4 | 0.3428 | 0.3456 | 87.38 |
| 5 | 0.3246 | 0.3433 | 87.68 |

Final test loss was 0.3428 and test accuracy was 87.85%.

### 5. Checkpoint and Resume

Checkpoint saving and loading were tested successfully. The resume run loaded the saved checkpoint and reported "Resuming from epoch 6". Since the configured training consisted of 5 epochs, all epochs had already been completed, so no additional training was necessary.

### 6. Additional SGD Experiment

A separate SGD run with batch size 64, gradient accumulation steps 4, and 5 epochs achieved:

- Test loss: 0.6045
- Test accuracy: 78.48%

This was a separate experiment from the 10-epoch optimizer and scheduler comparison.

### 7. Profiling

A PyTorch profiler trace was generated and saved to:

`results/profile_trace.json`

The trace can be inspected using a compatible profiling tool.

## Conclusions

1. Adam outperformed SGD in the 10-epoch optimizer and scheduler comparison. Adam with CosineAnnealingLR achieved the highest test accuracy of 89.49%.
2. Batch size affects the trade-off between throughput and accuracy. Batch size 512 achieved the highest throughput, while batch size 32 achieved the highest validation accuracy.
3. Data loading benefited from parallel workers. The best measured throughput was achieved with num_workers = 2. Increasing the worker count further did not provide additional benefits.

## Project Structure

- `src/` — training and benchmark scripts.
- `configs/` — YAML configuration files.
- `checkpoints/` — saved model checkpoints.
- `results/` — metrics, benchmark CSV files, graph, and profiler trace.
