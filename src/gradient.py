from copy import deepcopy
from typing import List, Callable
from partial import partial

type Vector = List[float]
type Matrix = List[List[float]]


def gradient(
    f: Callable[[Vector], float],
    x: Vector,
    h: float = 0.001,
):
    return [partial(f, x, i) for i, _ in enumerate(x)]


def grad(f: Callable[[Vector], float], x: Vector, h: float = 0.001):
    return gradient(f, x, h)


# Not sure about this one
# def nabla(
#     f: Callable[[Vector], float],
#     x: Vector,
#     h: float = 0.001,
# ) -> float:
#     return grad(f, x, h)
