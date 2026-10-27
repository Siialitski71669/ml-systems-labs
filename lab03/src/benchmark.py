import sys
import csv
import time
from pathlib import Path
from statistics import median

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab03 import MLP, CNN, AttentionModel

out = Path(__file__).resolve().parents[1] / "results" / "inference_latency.csv"
device = torch.device("cpu")
models = {"MLP": MLP(), "CNN": CNN(), "Attention": AttentionModel()}
rows = []

for name, model in models.items():
    model.eval()
    for batch_size, repeats in [(1, 100), (128, 5)]:
        x = torch.randn(batch_size, 1, 28, 28)
        with torch.no_grad():
            for _ in range(10):
                model(x)
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                model(x)
                times.append((time.perf_counter() - start) * 1000)
        rows.append({
            "model": name,
            "batch_size": batch_size,
            "median_latency_ms": round(median(times), 4),
            "repeats": repeats,
        })
        print(name, "batch =", batch_size, "repeats =", repeats,
              "median ms =", round(median(times), 4), flush=True)

with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", out)
