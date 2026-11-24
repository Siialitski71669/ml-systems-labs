# Lab 07 — Compression: Pruning and Knowledge Distillation

This folder is designed to be copied into the root of `ml-systems-labs`.

## What it does

`run_all.py` runs the complete experiment on Fashion-MNIST:

1. Train an oversized teacher CNN (target: about 1M parameters).
2. Apply global and layer-wise magnitude pruning at sparsities 0, 0.25, 0.5, 0.75, 0.9, 0.95; fine-tune each model for 3 epochs with masks fixed.
3. Apply structured filter pruning at 10%, 25%, and 50%; physically build smaller dense CNNs and fine-tune them.
4. Compare dense and sparse-format checkpoint sizes and measure CPU inference latency.
5. Train a compact student CNN with ordinary cross-entropy and with knowledge distillation (T=4, alpha=0.5).
6. Save CSV files, plots, checkpoints, and an automatically populated report.

## Setup (PowerShell)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` already exists, install the Lab 7 dependencies with:

```powershell
pip install torch torchvision numpy pandas matplotlib PyYAML
```

## Run

```powershell
python lab07\run_all.py
```

Fashion-MNIST is downloaded automatically by torchvision on the first run. Internet access is required for that first download.

To change epochs, batch size, or number of workers, edit `lab07/configs/config.yaml`.

## Important measurement notes

- Unstructured pruning keeps the original dense tensor shapes. It reduces the number of nonzero weights, but standard dense CPU convolution/linear kernels may not become faster.
- Sparse-format size is measured by serializing nonzero values and their indices; it is not the same as the regular dense `state_dict` checkpoint size.
- CPU latency is measured with warm-up and repeated inference runs. Results depend on the computer, CPU load, PyTorch build, and thread count.
- Energy is reported as a transparent proxy (CPU inference time × configured nominal CPU power), not as a hardware energy-meter reading. Replace it with RAPL or an external power meter if available.
- The report is generated from actual run outputs. Do not compare timings from different computers as if they were measured under identical conditions.

## Output

- `results/pruning_results.csv`
- `results/structured_pruning_results.csv`
- `results/model_comparison.csv`
- `results/efficiency_pillars.csv`
- `results/pruning_accuracy.png`
- `results/model_tradeoffs.png`
- `results/report_generated.md`
- `checkpoints/` — trained models

The run can take a while on CPU because each pruning configuration is fine-tuned for 3 epochs.
