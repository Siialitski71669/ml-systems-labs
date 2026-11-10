from engine import Value
import math

x = Value(0.0)
y = x.sigmoid()
y.backward()

expected = 0.5
expected_grad = 0.25

assert abs(y.data - expected) < 1e-10
assert abs(x.grad - expected_grad) < 1e-10

print("Sigmoid: PASSED")
print(f"Output: {y.data}")
print(f"Gradient: {x.grad}")
print("ALL SIGMOID TESTS PASSED")
