from collections.abc import Callable

from derivative import d


def sign(x: float) -> float:
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0


def newton_raphson(
    f: Callable[[float], float], x_0: float, max_steps: int = 1_000
) -> float:
    """
    ⚠️ `f(x)` needs to be differentiable.
    """
    zero = x_0
    for _ in range(max_steps):
        zero -= f(zero) / d(f, zero)
    return zero


def secant(
    f: Callable[[float], float], x_0: float, x_1: float, max_steps: int = 1_000
) -> float:
    """
    This method does not require `f(x)` to be differentiable but instead requires 2 starting values.
    """
    prev1 = x_1
    prev2 = x_0
    zero = 0
    for _ in range(max_steps):
        # print(f"{zero = }, {prev1 = }, {prev2 = }")
        zero = (prev2 * f(prev1) - prev1 * f(prev2)) / (f(prev1) - f(prev2))
        prev1 = prev2
        prev2 = zero
        # Early return to prevent division by zero.
        if prev2 == prev1:
            return zero
    return zero


def bisect(
    f: Callable[[float], float], x_0: float, x_1: float, max_steps: int = 1_000
) -> float:
    """
    If we have 2 points `a` and `b` and the signs of `f(a)` and `f(b)` are not the same, then `f`
    must at some point cross thru the x axis. This point is then found using binary search.
    """
    assert sign(f(x_0)) != sign(f(x_1))

    # f(a) < 0 and f(b) > 0
    a, b = 0, 0
    if sign(f(x_0)) == -1 and sign(f(x_1)) == 1:
        a, b = x_0, x_1
    else:
        a, b = x_1, x_0

    c = 0

    for _ in range(max_steps):
        c = (a + b) / 2
        if f(c) == 0:
            return c
        if sign(f(c)) == -1:
            a = c
        else:
            b = c

    return c
