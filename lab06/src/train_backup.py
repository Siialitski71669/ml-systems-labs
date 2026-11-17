import argparse
import csv
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models import CNN

LAB_DIR = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_data_loaders(config):
    data_config = config["dataset"]
    training = config["training"]
    data_dir = (LAB_DIR / data_config["data_dir"]).resolve()

    transform = transforms.ToTensor()
    full_train = datasets.FashionMNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_data = datasets.FashionMNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    train_data, validation_data = random_split(
        full_train,
        [data_config["train_size"], data_config["validation_size"]],
        generator=torch.Generator().manual_seed(
            config["reproducibility"]["seed"]
        ),
    )

    generator = torch.Generator().manual_seed(
        config["reproducibility"]["seed"]
    )

    train_loader = DataLoader(
        train_data,
        batch_size=training["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=training["num_workers"],
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=training["batch_size"],
        shuffle=False,
        num_workers=training["num_workers"],
    )
    test_loader = DataLoader(
        test_data,
        batch_size=training["batch_size"],
        shuffle=False,
        num_workers=training["num_workers"],
    )

    return train_loader, validation_loader, test_loader


def create_optimizer(model, config):
    training = config["training"]
    name = training["optimizer"].lower()

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=training["learning_rate"],
            momentum=training["momentum"],
            weight_decay=training["weight_decay"],
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def create_scheduler(optimizer, config):
    training = config["training"]
    name = training["scheduler"].lower()
    epochs = training["epochs"]

    if name == "none":
        return None
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.5
        )
    if name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    if name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def save_checkpoint(path, epoch, model, optimizer, scheduler, history):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "history": history,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])

    return checkpoint["epoch"], checkpoint["history"]


def save_metrics(path, history):
    if history:
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(LAB_DIR / "configs" / "config.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(
        config["reproducibility"]["seed"],
        config["reproducibility"]["deterministic"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, validation_loader, test_loader = create_data_loaders(config)

    model = CNN(
        input_channels=config["model"]["input_channels"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    checkpoint_dir = LAB_DIR / config["checkpoint"]["directory"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / config["checkpoint"]["filename"]

    results_dir = LAB_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "training_metrics.csv"

    start_epoch = 0
    history = []

    if config["checkpoint"]["resume"]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        start_epoch, history = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
        print(f"Resuming from epoch {start_epoch + 1}")

    tracking_uri = config["experiment"]["tracking_uri"]
    if tracking_uri.startswith("file:"):
        tracking_path = (
            LAB_DIR / tracking_uri.replace("file:", "", 1)
        ).resolve()
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = tracking_path.as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["experiment"]["name"])

    profiler_path = LAB_DIR / config["profiling"]["output_file"]
    profiler_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="CNN-training"):
        mlflow.log_params({
            "model": config["model"]["name"],
            "dataset": config["dataset"]["name"],
            "seed": config["reproducibility"]["seed"],
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "num_workers": config["training"]["num_workers"],
            "learning_rate": config["training"]["learning_rate"],
            "optimizer": config["training"]["optimizer"],
            "scheduler": config["training"]["scheduler"],
            "weight_decay": config["training"]["weight_decay"],
            "device": str(device),
        })

        mlflow.log_artifact(args.config)

        for epoch in range(start_epoch, config["training"]["epochs"]):
            model.train()
            epoch_start = time.perf_counter()
            loss_sum = 0.0
            profiler = None

            if config["profiling"]["enabled"] and epoch == start_epoch:
                from torch.profiler import profile, ProfilerActivity

                profiler = profile(
                    activities=[ProfilerActivity.CPU],
                    record_shapes=True,
                    profile_memory=True,
                )
                profiler.start()

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * labels.size(0)

                if profiler is not None:
                    profiler.step()

            if profiler is not None:
                profiler.stop()
                profiler.export_chrome_trace(str(profiler_path))
                print("Profiler trace saved:", profiler_path)

            train_loss = loss_sum / len(train_loader.dataset)
            val_loss, val_accuracy = evaluate(
                model, validation_loader, criterion, device
            )

            if scheduler is not None:
                if isinstance(
                    scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            learning_rate = optimizer.param_groups[0]["lr"]
            epoch_time = time.perf_counter() - epoch_start

            row = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "validation_loss": round(val_loss, 6),
                "validation_accuracy": round(val_accuracy, 4),
                "learning_rate": learning_rate,
                "epoch_time_seconds": round(epoch_time, 3),
            }
            history.append(row)

            print(
                f"Epoch {epoch + 1}/{config['training']['epochs']} "
                f"| train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} "
                f"| val_accuracy={val_accuracy:.2f}% "
                f"| time={epoch_time:.1f}s",
                flush=True,
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "learning_rate": learning_rate,
                "epoch_time_seconds": epoch_time,
            }, step=epoch + 1)

            if config["checkpoint"]["save_every_epoch"]:
                save_checkpoint(
                    checkpoint_path,
                    epoch + 1,
                    model,
                    optimizer,
                    scheduler,
                    history,
                )

            if config["logging"]["save_metrics_csv"]:
                save_metrics(metrics_path, history)

        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, device
        )

        print(f"\nTest loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.2f}%")

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        })

        for artifact in (metrics_path, checkpoint_path, profiler_path):
            if artifact.exists():
                mlflow.log_artifact(str(artifact))

    print("\nTraining completed successfully.")
    print("Checkpoint:", checkpoint_path)
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()

