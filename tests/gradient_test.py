import sys
from pathlib import Path

sys.path.insert(0, (Path() / "src/").absolute().as_posix())

from gradient import grad, Vector


def f(x: Vector):
    return x[0] ** 2 - 3 * x[1]


x: Vector = [3, 2]

print(f"{[f'{n:.4f}' for n in grad(f, x)]}")
