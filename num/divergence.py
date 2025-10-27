from typing import List, Callable
from partial import partial, partial2

type Vector = List[float]
type Matrix = List[List[float]]


def divergence(
    f: List[Callable[[Vector], Vector]],
    x: Vector,
    h: float = 0.001,
) -> float:
    return sum([partial(f, x, i, h) for i, f in enumerate(f)])


def div(f: List[Callable[[Vector], Vector]], x: Vector, h: float = 0.001) -> float:
    return divergence(f, x, h)


def laplace(f: List[Callable[[Vector], Vector]], x: Vector, h: float = 0.001) -> float:
    return sum([partial2(f, x, i, h) for i, f in enumerate(f)])
