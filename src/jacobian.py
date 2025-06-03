from typing import List, Callable
from copy import deepcopy

type Vector = List[float]
type Matrix = List[List[float]]


def D(
    f: Callable[[Vector], Vector],
    x: Vector,
    h: float = 0.001,
) -> Matrix:
    # Idk why this works but it does, somehow…
    def partial_col(
        f: Callable[[Vector], Vector],
        x: Vector,
        i: int,
        h: float = 0.001,
    ) -> Vector:
        """
        This internal function extracts the partial derivative for a single dimension from all
        functions.
        """
        a, b, c, d = deepcopy(x), deepcopy(x), deepcopy(x), deepcopy(x)
        a[i] -= 2 * h
        b[i] -= h
        c[i] += h
        d[i] += 2 * h
        return [
            (a - 8 * b + 8 * c - d) / (12 * h)
            for a, b, c, d in zip(f(a), f(b), f(c), f(d))
        ]

    D = [partial_col(f, x, i, h) for i in range(len(x))]

    # Transposing result
    return [[D[j][i] for j in range(len(D))] for i in range(len(D[0]))]
