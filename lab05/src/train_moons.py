import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_moons

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Value


N_SAMPLES = 120
EPOCHS = 20
LEARNING_RATE = 0.05
REPEATS = 5


def create_data():
    X, y = make_moons(
        n_samples=N_SAMPLES,
        noise=0.2,
        random_state=42
    )
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    return X, y


def initialize_weights():
    rng = np.random.default_rng(42)

    return {
        "w1": rng.normal(0, 0.3, (2, 8)),
        "b1": np.zeros(8),
        "w2": rng.normal(0, 0.3, 8),
        "b2": 0.0
    }


def custom_train(X, y, initial):
    w1 = [[Value(v) for v in row] for row in initial["w1"]]
    b1 = [Value(v) for v in initial["b1"]]
    w2 = [Value(v) for v in initial["w2"]]
    b2 = Value(initial["b2"])

    parameters = (
        [p for row in w1 for p in row]
        + b1 + w2 + [b2]
    )

    start = time.perf_counter()

    for epoch in range(EPOCHS):
        for xi, yi in zip(X, y):
            hidden = []

            for j in range(8):
                z = (
                    Value(xi[0]) * w1[0][j]
                    + Value(xi[1]) * w1[1][j]
                    + b1[j]
                )
                hidden.append(z.relu())

            output = b2

            for j in range(8):
                output = output + hidden[j] * w2[j]

            prediction = output.sigmoid()

            loss = -(
                yi * prediction.log()
                + (1.0 - yi) * (1.0 - prediction).log()
            )

            loss.backward()

            for p in parameters:
                p.data -= LEARNING_RATE * p.grad
                p.grad = 0.0

    elapsed = time.perf_counter() - start

    correct = 0
    for xi, yi in zip(X, y):
        hidden = []
        for j in range(8):
            z = (
                xi[0] * w1[0][j].data
                + xi[1] * w1[1][j].data
                + b1[j].data
            )
            hidden.append(max(0.0, z))

        output = b2.data + sum(
            hidden[j] * w2[j].data for j in range(8)
        )
        prediction = 1.0 / (1.0 + np.exp(-output))
        correct += int((prediction >= 0.5) == yi)

    return elapsed, correct / len(y)


def torch_train(X, y, initial):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
        torch.nn.Sigmoid()
    ).double()

    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(initial["w1"].T))
        model[0].bias.copy_(torch.tensor(initial["b1"]))
        model[2].weight.copy_(torch.tensor(initial["w2"].reshape(1, 8)))
        model[2].bias.copy_(torch.tensor([initial["b2"]]))

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    inputs = torch.tensor(X, dtype=torch.float64)
    targets = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)

    start = time.perf_counter()

    for epoch in range(EPOCHS):
        for i in range(len(X)):
            optimizer.zero_grad()
            prediction = model(inputs[i:i + 1])
            loss = criterion(prediction, targets[i:i + 1])
            loss.backward()
            optimizer.step()

    elapsed = time.perf_counter() - start

    with torch.no_grad():
        predictions = model(inputs).reshape(-1)
        accuracy = ((predictions >= 0.5) == targets.reshape(-1)).double().mean().item()

    return elapsed, accuracy


def main():
    X, y = create_data()

    custom_times = []
    torch_times = []

    for run in range(REPEATS):
        initial = initialize_weights()

        custom_time, custom_accuracy = custom_train(X, y, initial)
        torch_time, torch_accuracy = torch_train(X, y, initial)

        custom_times.append(custom_time)
        torch_times.append(torch_time)

        print(f"Run {run + 1}/{REPEATS} completed")

    custom_median = float(np.median(custom_times))
    torch_median = float(np.median(torch_times))

    print("\nTRAINING RESULTS")
    print(f"Dataset: make_moons, {N_SAMPLES} samples")
    print("Architecture: 2 -> 8 -> 1")
    print(f"Epochs: {EPOCHS}")
    print(f"Custom engine accuracy: {custom_accuracy:.4f}")
    print(f"PyTorch accuracy:       {torch_accuracy:.4f}")

    print("\nMEDIAN EXECUTION TIME (5 RUNS)")
    print(f"Custom engine: {custom_median:.4f} seconds")
    print(f"PyTorch:       {torch_median:.4f} seconds")
    print(f"Custom / PyTorch: {custom_median / torch_median:.2f}x")


if __name__ == "__main__":
    main()
