from typing import Callable


def d(
    f: Callable[[float], float],
    x: float,
    h: float = 0.001,
) -> float:
    """Calculates first order derivative of `f` at `x` with precision `h`."""
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)


def d2(
    f: Callable[[float], float],
    x: float,
    h: float = 0.001,
) -> float:
    """Calculates second order derivative of `f` at `x` with precision `h`."""
    return d(lambda x: d(f, x, h), x, h)
