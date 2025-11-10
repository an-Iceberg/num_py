from copy import deepcopy
from typing import List, Callable

type Vector = List[float]


def partial(
    f: Callable[[Vector], float],
    x: Vector,
    i: int,
    h: float = 0.001,
) -> float:
    x_vec = deepcopy(x)
    x_val = x_vec[i]

    def x(h: float) -> list[float]:
        x_vec[i] = x_val
        x_vec[i] += h
        return x_vec

    return (f(x(-2 * h)) - 8 * f(x(-h)) + 8 * f(x(h)) - f(x(2 * h))) / (12 * h)


def partial2(
    f: Callable[[Vector], float],
    x: Vector,
    i: int,
    h: float = 0.001,
) -> float:
    return partial(lambda x: partial(f, x, i, h), x, i, h)


def partial3(
    f: Callable[[Vector], float],
    x: Vector,
    i: int,
    h: float = 0.001,
) -> float:
    return partial(lambda x: partial2(f, x, i, h), x, i, h)


def partial4(
    f: Callable[[Vector], float],
    x: Vector,
    i: int,
    h: float = 0.001,
) -> float:
    return partial(lambda x: partial3(f, x, i, h), x, i, h)


def part(f: Callable[[Vector], float], x: Vector, i: int, h: float = 0.001):
    return partial(f, x, i, h)


def part2(f: Callable[[Vector], float], x: Vector, i: int, h: float = 0.001):
    return partial2(f, x, i, h)


def part3(f: Callable[[Vector], float], x: Vector, i: int, h: float = 0.001):
    return partial3(f, x, i, h)


def part4(f: Callable[[Vector], float], x: Vector, i: int, h: float = 0.001):
    return partial4(f, x, i, h)
