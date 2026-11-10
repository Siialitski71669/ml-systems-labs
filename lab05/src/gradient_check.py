import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Value


def custom_gradient(a_value, b_value, c_value):
    a = Value(a_value)
    b = Value(b_value)
    c = Value(c_value)

    result = ((a * b + c) ** 2).relu()
    result.backward()

    return result.data, np.array([a.grad, b.grad, c.grad])


def torch_gradient(a_value, b_value, c_value):
    a = torch.tensor(a_value, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(b_value, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(c_value, dtype=torch.float64, requires_grad=True)

    result = torch.relu((a * b + c) ** 2)
    result.backward()

    return result.item(), np.array([
        a.grad.item(),
        b.grad.item(),
        c.grad.item()
    ])


def numerical_gradient(values, epsilon=1e-6):
    gradients = []

    def function(x):
        a, b, c = x
        return max(0.0, (a * b + c) ** 2)

    for i in range(len(values)):
        plus = values.copy()
        minus = values.copy()

        plus[i] += epsilon
        minus[i] -= epsilon

        derivative = (
            function(plus) - function(minus)
        ) / (2 * epsilon)

        gradients.append(derivative)

    return np.array(gradients)


def main():
    values = [2.0, 3.0, 4.0]

    custom_output, custom_grad = custom_gradient(*values)
    torch_output, torch_grad = torch_gradient(*values)
    numeric_grad = numerical_gradient(values)

    print("OUTPUT COMPARISON")
    print(f"Custom engine: {custom_output}")
    print(f"PyTorch:       {torch_output}")

    print("\nGRADIENT COMPARISON")
    print(f"Custom engine: {custom_grad}")
    print(f"PyTorch:       {torch_grad}")
    print(f"Numerical:     {numeric_grad}")

    torch_diff = np.max(np.abs(custom_grad - torch_grad))
    numeric_diff = np.max(np.abs(custom_grad - numeric_grad))

    print("\nMAXIMUM ABSOLUTE DIFFERENCES")
    print(f"Custom vs PyTorch:   {torch_diff:.12f}")
    print(f"Custom vs Numerical: {numeric_diff:.12f}")

    assert torch_diff < 1e-6
    assert numeric_diff < 1e-6

    print("\nALL GRADIENT CHECKS PASSED")


if __name__ == "__main__":
    main()
