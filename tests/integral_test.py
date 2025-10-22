import sys
from pathlib import Path

sys.path.insert(0, (Path() / "src/").absolute().as_posix())

from math import sin, cos
from integral import int_2, int_3, int_4, int_6


def I(a: int, b: int):
    return cos(b) - cos(a)


# Todo: iterate over different ranges and compute the integrals
