import sys
from pathlib import Path

sys.path.insert(0, (Path() / "src/").absolute().as_posix())

from derivative import d, d1, d2
from math import sin, cos

for x in range(10):
    x /= 10
    x += 1
    print(f"d  ε = {abs(cos(x) - d(sin, x)):.2e}")
    print(f"d1 ε = {abs(cos(x) - d1(sin, x)):.2e}")
    print(f"d2 ε = {abs(-sin(x) - d2(sin, x)):.2e}")
    print(f"d2 alt ε = {abs(-sin(x) - d(lambda x: d1(sin, x), x)):.2e}")
    print()
