import sys
from pathlib import Path

sys.path.insert(0, (Path() / "src/").absolute().as_posix())

from jacobi import D, Vector, Matrix
from math import log, exp

x = [1.5, 3, 2.5]


def f1(x: Vector) -> float:
    x, y, z = x
    return x + y**2 - z**2 - 13


def f2(x: Vector) -> float:
    x, y, z = x
    return log(y / 4) + exp((0.5 * z) - 1) - 1


def f3(x: Vector) -> float:
    x, y, z = x
    return (y - 3) ** 2 - z**3 + 7


def D_test():
    import numpy as np
    from typing import Callable

    def D(
        f: Callable[[Vector], Vector],
        x: Vector,
        h: float = 0.001,
    ) -> Matrix:
        from copy import deepcopy

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

    def f(λ: Vector) -> Vector:
        return [
            f1(λ),
            f2(λ),
            f3(λ),
        ]

    print(np.array_str(np.array(D(f, x)), precision=5, suppress_small=True))


D_test()

print()


def D_new_test():
    import numpy as np

    f = [f1, f2, f3]
    print(np.array_str(np.array(D(f, x)), precision=5, suppress_small=True))


D_new_test()
