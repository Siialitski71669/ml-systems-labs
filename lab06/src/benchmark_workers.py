import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models import CNN

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
RESULTS = ROOT / "results"


def main():
    dataset = datasets.FashionMNIST(
        root=str(DATA), train=True, download=True,
        transform=transforms.ToTensor()
    )
    train_data, _ = random_split(
        dataset, [55000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    rows = []

    for workers in [0, 1, 2, 4, 8]:
        print(f"\nnum_workers={workers}", flush=True)

        loader = DataLoader(
            train_data, batch_size=128, shuffle=True,
            num_workers=workers
        )
        model = CNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        model.train()

        data_t = forward_t = backward_t = optimizer_t = 0.0
        samples = 0
        iterator = iter(loader)

        while True:
            t = time.perf_counter()
            try:
                images, labels = next(iterator)
            except StopIteration:
                break
            data_t += time.perf_counter() - t
            samples += len(labels)

            t = time.perf_counter()
            loss = criterion(model(images), labels)
            forward_t += time.perf_counter() - t

            t = time.perf_counter()
            loss.backward()
            backward_t += time.perf_counter() - t

            t = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_t += time.perf_counter() - t

        total = data_t + forward_t + backward_t + optimizer_t
        row = {
            "num_workers": workers,
            "samples": samples,
            "total_seconds": round(total, 2),
            "samples_per_second": round(samples / total, 2),
            "data_loading_percent": round(data_t / total * 100, 2),
            "forward_percent": round(forward_t / total * 100, 2),
            "backward_percent": round(backward_t / total * 100, 2),
            "optimizer_percent": round(optimizer_t / total * 100, 2),
        }
        rows.append(row)
        print(row, flush=True)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "worker_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", RESULTS / "worker_benchmark.csv")


if __name__ == "__main__":
    main()
