from typing import List, Callable
from partial import partial

type Vector = List[float]
type Matrix = List[List[float]]


def gradient(
    f: Callable[[Vector], float],
    x: Vector,
    h: float = 0.001,
) -> list[float]:
    return [partial(f, x, i, h) for i, _ in enumerate(x)]

    # Copy more efficient method from Julia: h = fill(h, length(x))
    # h_vec = [h] * len(x)

    # def add(list_1: list[float], list_2: list[float]) -> list[float]:
    #     return [a + b for a, b in zip(list_1, list_2)]

    # (f(x - 2h) - 8f(x - h) + 8f(x + h) - f(x + 2h)) / (12 * h)
    # return (
    #     f([x - 2 * h for x, h in zip(x, h_vec)])
    #     - 8 * f([x - h for x, h in zip(x, h_vec)])
    #     + 8 * f([x - h for x, h in zip(x, h_vec)])
    #     - f([x - 2 * h for x, h in zip(x, h_vec)])
    # ) / (12 * h)


def grad(f: Callable[[Vector], float], x: Vector, h: float = 0.001):
    return gradient(f, x, h)


def nabla(f: Callable[[Vector], float], x: Vector, h: float = 0.001):
    return gradient(f, x, h)


# Not sure about this one
# def nabla(
#     f: Callable[[Vector], float],
#     x: Vector,
#     h: float = 0.001,
# ) -> float:
#     return grad(f, x, h)
