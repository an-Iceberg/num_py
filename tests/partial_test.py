import sys
from pathlib import Path

sys.path.insert(0, (Path() / "src/").absolute().as_posix())

from partial import Vector, partial


def f(x: Vector):
    return x[0] ** 2 - 3 * x[1]


x: Vector = [3, 2]

print(f"{partial(f, x, 0):.4f}")
print(f"{partial(f, x, 1):.4f}")
