from collections.abc import Callable


def d(f: Callable[[float], float], x: float, h: float = 0.0001) -> float:
    """Calculates first order derivative of `f` at `x` with precision `h` using the [5 point stencil](https://en.wikipedia.org/wiki/Five-point_stencil) (the middle point is multiplied by a factor of 0 so it's effectively a 4 point stencil). """
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)


def d2(f: Callable[[float], float], x: float, h: float = 0.0001) -> float:
    """Calculates second order derivative of `f` at `x` with precision `h`."""
    return d(lambda x: d(f, x, h), x, h)


def d3(f: Callable[[float], float], x: float, h: float = 0.0001) -> float:
    """Calculates third order derivative of `f` at `x` with precision `h`."""
    return d2(lambda x: d(f, x, h), x, h)


def d4(f: Callable[[float], float], x: float, h: float = 0.0001) -> float:
    """Calculates forth order derivative of `f` at `x` with precision `h`."""
    return d3(lambda x: d(f, x, h), x, h)
