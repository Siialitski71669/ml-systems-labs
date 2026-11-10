from engine import Value


def test_main_expression():
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)

    result = ((a * b + c) ** 2).relu()
    result.backward()

    assert result.data == 100.0
    assert a.grad == 60.0
    assert b.grad == 40.0
    assert c.grad == 20.0

    print("Main expression: PASSED")
    print(f"Output: {result.data}")
    print(f"da: {a.grad}, db: {b.grad}, dc: {c.grad}")


def test_gradient_accumulation():
    x = Value(3.0)
    result = x * x + x
    result.backward()

    assert result.data == 12.0
    assert x.grad == 7.0

    print("Gradient accumulation: PASSED")
    print(f"Expected gradient: 7.0, actual: {x.grad}")


def test_relu():
    x = Value(-5.0)
    result = x.relu()
    result.backward()

    assert result.data == 0.0
    assert x.grad == 0.0

    print("ReLU: PASSED")


if __name__ == "__main__":
    test_main_expression()
    test_gradient_accumulation()
    test_relu()
    print("\nALL TESTS PASSED")
