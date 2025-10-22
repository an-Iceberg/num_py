from jacobi import D
from math import exp, log
import numpy as np


def f1(x):
    x, y, z = x
    return x + y**2 - z**2 - 13


def f2(x):
    _, y, z = x
    return log(y / 4) + exp(0.5 * z - 1) - 1


def f3(x):
    _, y, z = x
    return (y - 3) ** 2 - z**3 + 7


f = [f1, f2, f3]


x = [1.5, 3, 2.5]

print(np.matrix(D(f, x)))
