from math import sin, cos, pi as π
from partial import partial, partial2


def f(input):
    x, y = input
    return sin(x) + cos(y)


def dx_f(input):
    x, _ = input
    return cos(x)


def dy_f(input):
    _, y = input
    return -sin(y)


def d2x_f(input):
    x, _ = input
    return -sin(x)


def d2y_f(input):
    _, y = input
    return -cos(y)


x = [π, π]

print(f"dx ε = {abs(dx_f(x) - partial(f, x, 0)):.2e}")
print(f"dy ε = {abs(dy_f(x) - partial(f, x, 1)):.2e}\n")

print(f"d2x ε = {abs(d2x_f(x) - partial2(f, x, 0)):.2e}")
print(f"d2y ε = {abs(d2y_f(x) - partial2(f, x, 1)):.2e}")
