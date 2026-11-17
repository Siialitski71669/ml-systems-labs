import csv
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn

from train import (
    LAB_DIR,
    CNN,
    load_config,
    set_seed,
    create_data_loaders,
    create_optimizer,
    create_scheduler,
    evaluate,
)


def main():
    config_path = LAB_DIR / "configs" / "config.yaml"
    config = load_config(config_path)

    config["training"]["epochs"] = 10
    config["training"]["batch_size"] = 128
    config["training"]["gradient_accumulation_steps"] = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    combinations = [
        ("Adam", "StepLR"),
        ("Adam", "CosineAnnealingLR"),
        ("Adam", "ReduceLROnPlateau"),
        ("SGD", "StepLR"),
        ("SGD", "CosineAnnealingLR"),
        ("SGD", "ReduceLROnPlateau"),
    ]

    results_dir = LAB_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "optimizer_scheduler_comparison.csv"

    tracking_uri = config["experiment"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Lab06-Optimizer-Scheduler-Comparison")

    results = []

    print(f"Device: {device}")
    print("Starting 6 experiments, 10 epochs each.")

    for index, (optimizer_name, scheduler_name) in enumerate(combinations, 1):
        print(
            f"\n[{index}/6] {optimizer_name} + {scheduler_name}",
            flush=True,
        )

        set_seed(config["reproducibility"]["seed"])

        config["training"]["optimizer"] = optimizer_name
        config["training"]["scheduler"] = scheduler_name

        train_loader, val_loader, test_loader = create_data_loaders(config)

        model = CNN(
            input_channels=config["model"]["input_channels"],
            num_classes=config["model"]["num_classes"],
        ).to(device)

        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)

        start_time = time.time()
        final_val_loss = 0.0
        final_val_accuracy = 0.0

        with mlflow.start_run(
            run_name=f"{optimizer_name}_{scheduler_name}"
        ):
            mlflow.log_params({
                "optimizer": optimizer_name,
                "scheduler": scheduler_name,
                "epochs": 10,
                "batch_size": 128,
                "learning_rate": config["training"]["learning_rate"],
                "momentum": config["training"]["momentum"],
                "seed": config["reproducibility"]["seed"],
                "device": str(device),
            })
            mlflow.log_artifact(str(config_path))

            for epoch in range(10):
                model.train()
                running_loss = 0.0
                samples = 0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * labels.size(0)
                    samples += labels.size(0)

                train_loss = running_loss / samples
                final_val_loss, final_val_accuracy = evaluate(
                    model, val_loader, criterion, device
                )

                if scheduler_name == "ReduceLROnPlateau":
                    scheduler.step(final_val_loss)
                else:
                    scheduler.step()

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "validation_loss": final_val_loss,
                    "validation_accuracy": final_val_accuracy,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }, step=epoch + 1)

                print(
                    f"Epoch {epoch + 1}/10 | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_acc={final_val_accuracy:.2f}%",
                    flush=True,
                )

            test_loss, test_accuracy = evaluate(
                model, test_loader, criterion, device
            )
            elapsed = time.time() - start_time

            mlflow.log_metrics({
                "final_test_loss": test_loss,
                "final_test_accuracy": test_accuracy,
                "training_seconds": elapsed,
            })

            results.append({
                "optimizer": optimizer_name,
                "scheduler": scheduler_name,
                "epochs": 10,
                "batch_size": 128,
                "final_validation_accuracy": round(final_val_accuracy, 4),
                "final_test_accuracy": round(test_accuracy, 4),
                "training_seconds": round(elapsed, 2),
            })

            print(
                f"Finished: test accuracy={test_accuracy:.2f}%, "
                f"time={elapsed:.1f}s",
                flush=True,
            )

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with mlflow.start_run(run_name="Comparison-Summary"):
        mlflow.log_artifact(str(output_path))

    print("\nAll experiments completed.")
    print(f"Results saved to: {output_path}")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
