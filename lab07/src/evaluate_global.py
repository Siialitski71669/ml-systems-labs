import sys
from pathlib import Path
import yaml
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import make_teacher, count_parameters
from engine import set_seed, make_loaders, evaluate, load_checkpoint, count_nonzero_parameters, measure_latency

def main():
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["root"] = str(ROOT / "data")
    set_seed(cfg["seed"])

    device = torch.device("cpu")
    _, _, test_loader = make_loaders(cfg)

    ckpts = ROOT / "checkpoints"
    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    bench = cfg["benchmark"]
    rows = []

    for amount in [0, 25, 50, 75, 90, 95]:
        name = f"global_{amount}"
        checkpoint = ckpts / f"{name}_dense.pth"
        sparse_file = ckpts / f"{name}_sparse.npz"

        if not checkpoint.exists() or not sparse_file.exists():
            raise FileNotFoundError(f"Missing checkpoint: {name}")

        model = make_teacher().to(device)
        load_checkpoint(model, checkpoint, device)

        _, accuracy = evaluate(model, test_loader, device)
        latency, _ = measure_latency(
            model, test_loader, device,
            bench["warmup_batches"], bench["repeats"], bench["cpu_threads"]
        )

        nonzero, total = count_nonzero_parameters(model)
        dense_bytes = checkpoint.stat().st_size
        sparse_bytes = sparse_file.stat().st_size
        actual_sparsity = 1.0 - nonzero / total
        energy = latency * float(bench["nominal_cpu_power_watts"])

        row = {
            "model": f"global {amount}%",
            "family": "global",
            "sparsity": actual_sparsity,
            "accuracy": accuracy,
            "parameters": count_parameters(model),
            "nonzero_parameters": nonzero,
            "latency_ms_per_image": latency,
            "dense_size_bytes": dense_bytes,
            "sparse_format_bytes": sparse_bytes,
            "target_filter_removal": "",
            "estimated_energy_mj_per_image": energy,
        }
        rows.append(row)

        print(
            f'{name}: accuracy={accuracy:.4%}, '
            f'sparsity={actual_sparsity:.2%}, latency={latency:.4f} ms/image'
        )

    global_df = pd.DataFrame(rows)
    global_df.to_csv(results / "global_pruning_results.csv", index=False)

    comparison_path = results / "model_comparison.csv"
    existing = pd.read_csv(comparison_path)
    existing = existing[existing["family"] != "global"]
    combined = pd.concat([existing, global_df], ignore_index=True)
    combined.to_csv(comparison_path, index=False)

    print("\nSaved global_pruning_results.csv")
    print("Updated model_comparison.csv")

if __name__ == "__main__":
    main()
