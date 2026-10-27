import csv
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from lab03 import MLP, CNN, AttentionModel

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_all = datasets.FashionMNIST(
        root=BASE / "data", train=True, download=True, transform=transform
    )
    test_data = datasets.FashionMNIST(
        root=BASE / "data", train=False, download=True, transform=transform
    )

    train_data, val_data = random_split(
        train_all, [55000, 5000],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    rows = []

    for name, model in models.items():
        print(f"\n=== {name} ===")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            model.train()
            loss_sum = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * labels.size(0)

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    predictions = model(images).argmax(dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"loss={loss_sum / len(train_data):.4f} | "
                f"validation accuracy={100 * correct / total:.2f}%"
            )

        model.eval()
        correct = total = 0
        start = time.perf_counter()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        elapsed = time.perf_counter() - start
        accuracy = 100 * correct / total
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        rows.append({
            "model": name,
            "parameters": params,
            "test_accuracy_percent": round(accuracy, 2),
            "test_inference_seconds": round(elapsed, 4),
        })
        print(f"Test accuracy: {accuracy:.2f}%")
        print(f"Trainable parameters: {params:,}")

    with (RESULTS / "training_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved results to:", RESULTS / "training_results.csv")


if __name__ == "__main__":
    main()
