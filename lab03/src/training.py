import csv
import random
import time
import sys
from pathlib import Path
from statistics import median

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lab03 import MLP, CNN, AttentionModel

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT.parent / "data"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
LR = 0.001
REPEATS = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Device:", device)
print("Seed:", SEED)

transform = transforms.ToTensor()
train_all = datasets.FashionMNIST(
    root=str(DATA), train=True, download=True, transform=transform
)
test_data = datasets.FashionMNIST(
    root=str(DATA), train=False, download=True, transform=transform
)

train_data, val_data = random_split(
    train_all, [55000, 5000],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(SEED), num_workers=0
)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel(),
}

training_rows = []
cost_rows = []
latency_rows = []

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

def measure_latency(model, batch_size):
    sample = torch.randn(batch_size, 1, 28, 28, device=device)
    model.eval()
    with torch.no_grad():
        model(sample)  # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
        measurements = []
        for _ in range(REPEATS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            measurements.append((time.perf_counter() - start) * 1000)
    return median(measurements)

for name, model in models.items():
    print(f"\n===== {name} =====", flush=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)

        val_accuracy = evaluate(model, val_loader)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"loss={loss_sum / len(train_data):.4f} "
            f"val_accuracy={val_accuracy:.2f}%",
            flush=True
        )

    test_accuracy = evaluate(model, test_loader)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    training_rows.append({
        "model": name,
        "parameters": params,
        "test_accuracy_percent": round(test_accuracy, 2),
    })
    print(f"Test accuracy: {test_accuracy:.2f}%", flush=True)

    # Estimate MACs using THOP when available.
    macs = None
    try:
        from thop import profile
        macs, _ = profile(
            model, inputs=(torch.zeros(1, 1, 28, 28, device=device),),
            verbose=False
        )
        macs = int(macs)
    except Exception as exc:
        print(f"MAC profiling unavailable for {name}: {exc}")

    cost_rows.append({
        "model": name,
        "parameters": params,
        "MACs": macs if macs is not None else "",
        "approx_FLOPs": int(macs * 2) if macs is not None else "",
    })

    latency_rows.append({
        "model": name,
        "batch_size": 1,
        "median_latency_ms": round(measure_latency(model, 1), 4),
        "repeats": REPEATS,
    })
    latency_rows.append({
        "model": name,
        "batch_size": BATCH_SIZE,
        "median_latency_ms": round(measure_latency(model, BATCH_SIZE), 4),
        "repeats": REPEATS,
    })

def save_csv(filename, rows):
    with (RESULTS / filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

save_csv("training_results.csv", training_rows)
save_csv("model_costs.csv", cost_rows)
save_csv("inference_latency.csv", latency_rows)

plt.figure()
plt.bar(
    [r["model"] for r in training_rows],
    [r["test_accuracy_percent"] for r in training_rows]
)
plt.ylabel("Test accuracy (%)")
plt.title("Fashion-MNIST: Accuracy by Architecture")
plt.tight_layout()
plt.savefig(RESULTS / "accuracy_vs_parameters.png", dpi=150)
plt.close()

plt.figure()
for name in models:
    rows = [r for r in latency_rows if r["model"] == name]
    plt.plot(
        [r["batch_size"] for r in rows],
        [r["median_latency_ms"] for r in rows],
        marker="o", label=name
    )
plt.xlabel("Batch size")
plt.ylabel("Median inference latency (ms)")
plt.title("Inference Latency by Architecture")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "inference_latency.png", dpi=150)
plt.close()

print("\nAll results saved to:", RESULTS)
print("Training results:", training_rows)
print("Cost results:", cost_rows)
print("Latency results:", latency_rows)
