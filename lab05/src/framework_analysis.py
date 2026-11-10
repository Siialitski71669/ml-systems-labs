import time
import statistics

import torch
import torch.nn as nn


torch.manual_seed(42)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def benchmark(function, repeats=5, iterations=200):
    for _ in range(20):
        function()

    times = []

    for _ in range(repeats):
        start = time.perf_counter()

        for _ in range(iterations):
            function()

        times.append(time.perf_counter() - start)

    return statistics.median(times) / iterations


def main():
    model = SmallModel().eval()
    example = torch.randn(64, 16)

    with torch.no_grad():
        traced_model = torch.jit.trace(model, example)

        eager_output = model(example)
        traced_output = traced_model(example)

    print("MODEL GRAPH")
    print(traced_model.inlined_graph)

    difference = (eager_output - traced_output).abs().max().item()

    print("\nEAGER VS TRACED")
    print(f"Maximum output difference: {difference:.12f}")

    eager_time = benchmark(lambda: model(example))
    traced_time = benchmark(lambda: traced_model(example))

    print(f"Eager median per call:  {eager_time * 1000:.4f} ms")
    print(f"Traced median per call: {traced_time * 1000:.4f} ms")
    print(f"Eager / traced: {eager_time / traced_time:.2f}x")

    print("\nTENSOR PROPERTIES")

    base = torch.randn(1024, 1024)
    contiguous = base.contiguous()
    non_contiguous = base.t()

    for name, tensor in [
        ("Contiguous", contiguous),
        ("Transposed", non_contiguous)
    ]:
        print(f"\n{name}:")
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")
        print(f"Contiguous: {tensor.is_contiguous()}")
        print(f"Stride: {tensor.stride()}")

    contiguous_time = benchmark(
        lambda: (contiguous * 2 + 1).sum(),
        iterations=100
    )

    non_contiguous_time = benchmark(
        lambda: (non_contiguous * 2 + 1).sum(),
        iterations=100
    )

    print("\nCONTIGUOUS VS NON-CONTIGUOUS")
    print(f"Contiguous median:     {contiguous_time * 1000:.4f} ms")
    print(f"Non-contiguous median: {non_contiguous_time * 1000:.4f} ms")
    print(f"Non-contiguous / contiguous: {non_contiguous_time / contiguous_time:.2f}x")

    assert difference < 1e-6
    print("\nALL FRAMEWORK CHECKS PASSED")


if __name__ == "__main__":
    main()

