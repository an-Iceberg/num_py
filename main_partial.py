from math import sin, cos, pi as π
from copy import deepcopy


def partial(f, x_input, i, h=0.001):
    x_local = deepcopy(x_input)

    def x(h):
        x_local[i] = x_input[i]
        x_local[i] += h
        return x_local

    return (f(x(-2 * h)) - 8 * f(x(-h)) + 8 * f(x(h)) - f(x(2 * h))) / (12 * h)


def partial2(f, x, i, h=0.001):
    return partial(lambda x: partial(f, x, i, h), x, i, h)


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
