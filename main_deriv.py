from math import sin, cos, pi as π
from derivative import d, d2


def f(x):
    return sin(x)


def d_f(x):
    return cos(x)


def d2_f(x):
    return -sin(x)


x = π

print(f"d ε = {abs(d_f(x) - d(f, x)):.2e}")
print(f"d2 ε = {abs(d2_f(x) - d2(f, x)):.2e}")
