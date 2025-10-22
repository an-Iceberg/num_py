import sys
from pathlib import Path

sys.path.insert(0, (Path() / "src/").absolute().as_posix())

from divergence import div, laplace, Vector
from math import log, exp


def f1(x: Vector) -> float:
    x, y, z = x
    return x + y**2 - z**2 - 13


def f2(x: Vector) -> float:
    x, y, z = x
    return log(y / 4) + exp((0.5 * z) - 1) - 1


def f3(x: Vector) -> float:
    x, y, z = x
    return (y - 3) ** 2 - z**3 + 7


x = [1.5, 3, 2.5]
f = [f1, f2, f3]

# This is literally just summing the diagonal of the jacobi matrix
print(f"{div(f, x)}")
print(f"{sum([1, 0.333333333333333333, -18.75])}")
print(f"{laplace(f, x)}")
