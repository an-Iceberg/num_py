from typing import List
import num  # My own library for HM2
import math
import numpy as np
import matplotlib.pyplot as plt

type Vector = List[float]

A = 20
G = 450
K = 0.001
λ0 = [A, G, K]

ti = [0, 2, 4, 6, 8, 10]
Hi = [52.9, 184, 426, 529, 499, 510]


def H(t: float, coefficients: Vector) -> float:
    A, G, K = coefficients  # destructuring
    return G / (1 + (G / A - 1) * math.exp(-K * G * t))


# Umformung für das Gauss-Newton Verfahren
# g(x,y) = y - f(x)
def g(λ: Vector) -> Vector:
    return [y - H(x, λ) for x, y in zip(ti, Hi)]


coefficients, count = num.gauss_newton_damped(g, λ0, 1e-7, 100)

x = np.linspace(0, 10, 200)
y = [H(t, coefficients) for t in x]  # list comprehension

plt.scatter(ti, Hi, color="orange", zorder=1)
plt.xlabel("$t$")
plt.ylabel("$H(t)$")
plt.plot(x, y, zorder=0)
plt.show()
