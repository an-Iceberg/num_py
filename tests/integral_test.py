import sys
from pathlib import Path

sys.path.insert(0, (Path() / "src/").absolute().as_posix())

from math import sin, cos
from integral import int_2, int_3, int_4, int_6


def Int(a: float, b: float) -> float:
    return sin(b) - sin(a)


# Todo: iterate over different ranges and compute the integrals
b = 5.35
for a in range(10):
    a /= 10
    a += 2
    print(f"{abs(int_2(a, b, cos) - Int(a, b)):.2e}")
    print(f"{abs(int_3(a, b, cos) - Int(a, b)):.2e}")
    print(f"{abs(int_4(a, b, cos) - Int(a, b)):.2e}")
    print(f"{abs(int_6(a, b, cos) - Int(a, b)):.2e}")
    print()
