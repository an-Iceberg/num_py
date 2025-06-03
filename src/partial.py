from copy import deepcopy
from typing import List, Callable

type Vector = List[float]


def partial(
    f: Callable[[Vector], float],
    x_input: Vector,
    i: int,
    h: float = 0.001,
) -> float:
    x_local = deepcopy(x_input)
    x_val = x_local[i]

    def x(h):
        x_local[i] = x_val
        x_local[i] += h
        return x_local

    return (f(x(-2 * h)) - 8 * f(x(-h)) + 8 * f(x(h)) - f(x(2 * h))) / (12 * h)


def partial2(
    f: Callable[[Vector], float],
    x: Vector,
    i: int,
    h: float = 0.001,
) -> float:
    return partial(lambda x: partial(f, x, i, h), x, i, h)
