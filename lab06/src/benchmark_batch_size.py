import csv
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models import CNN

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
RESULTS = ROOT / "results"
DEVICE = torch.device("cpu")


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    dataset = datasets.FashionMNIST(
        root=str(DATA), train=True, download=True,
        transform=transforms.ToTensor()
    )
    train_data, val_data = random_split(
        dataset, [55000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    test_data = datasets.FashionMNIST(
        root=str(DATA), train=False, download=True,
        transform=transforms.ToTensor()
    )

    process = psutil.Process()
    rows = []

    for batch_size in [32, 64, 128, 256, 512]:
        print(f"\nBatch size={batch_size}", flush=True)

        torch.manual_seed(42)
        train_loader = DataLoader(
            train_data, batch_size=batch_size,
            shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_data, batch_size=512,
            shuffle=False, num_workers=0
        )

        model = CNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        peak_memory = process.memory_info().rss
        start = time.perf_counter()
        total_samples = 0

        for epoch in range(5):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

                total_samples += len(labels)
                peak_memory = max(
                    peak_memory, process.memory_info().rss
                )

            print(f"Epoch {epoch + 1}/5 completed", flush=True)

        elapsed = time.perf_counter() - start

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100

        row = {
            "batch_size": batch_size,
            "epochs": 5,
            "training_seconds": round(elapsed, 2),
            "samples_per_second": round(total_samples / elapsed, 2),
            "peak_memory_mb": round(peak_memory / (1024 ** 2), 2),
            "validation_accuracy_percent": round(accuracy, 2),
        }
        rows.append(row)
        print(row, flush=True)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "batch_size_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", RESULTS / "batch_size_benchmark.csv")


if __name__ == "__main__":
    main()
