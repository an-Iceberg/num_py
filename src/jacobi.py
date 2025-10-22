from typing import List, Callable
from partial import partial

type Vector = List[float]
type Matrix = List[List[float]]


def D(
    f: List[Callable[[Vector], float]],
    x: Vector,
    h: float = 0.001,
) -> Matrix:
    return [[partial(f, x, i) for i, _ in enumerate(x)] for f in f]


def D2(f: List[Callable[[Vector], float]], x: Vector, h: float = 0.001) -> Matrix:
    return D(lambda f: D(f, x, h), x, h)


def D3(f: List[Callable[[Vector], float]], x: Vector, h: float = 0.001) -> Matrix:
    return D(lambda f: D2(f, x, h), x, h)


def D4(f: List[Callable[[Vector], float]], x: Vector, h: float = 0.001) -> Matrix:
    return D(lambda f: D3(f, x, h), x, h)
