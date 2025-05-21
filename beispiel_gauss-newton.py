from typing import List
import num  # My own library for HM2
import math
import numpy as np
import matplotlib.pyplot as plt

type Vector = List[float]

# Beispiel 6.10 aus dem Skript

# Datenpunkte
x = [0, 1, 2, 3, 4]
y = [3, 1, 0.5, 0.2, 0.05]


# Formel, die an die Punkte angepasst werden soll
# y = ae^(bx)
def f(x: float, coefficients: Vector) -> float:
    a, b = coefficients
    return a * math.exp(b * x)


# TODO: move this part into the gauss-newton function body
# Umformung für das Gauss-Newton Verfahren
# g(a,b) = y - ae^(bx)
def g(λ: Vector) -> Vector:
    return [y - f(x, λ) for x, y in zip(x, y)]


ε = 1e-7

λ_1 = [1, -1.5]
λ_2 = [2.0, 2.0]
max_iter_count = 100

# Ungedämpft
λ, count = num.gauss_newton_print(g, λ_2, ε, max_iter_count)
a, b = λ
a = round(a, 4)
b = round(b, 4)
print("λ =", [round(n, 4) for n in λ])
print(f"count = {count}")
print(f"f(x) = {a}·e^({b}·x)")
print()

# Gedämpft
λ, count = num.gauss_newton_damped(g, λ_1, ε, max_iter_count)
a, b = λ
a = round(a, 4)
b = round(b, 4)
print("λ =", [round(n, 4) for n in λ])
print(f"count = {count}")
print(f"f(x) = {a}·e^({b}·x)")
print()

# Plotten des gedämpften Resultates
x_axis = np.linspace(0, 4, 200)
y_axis = [f(x, λ) for x in x_axis]

plt.scatter(x, y, color="orange", zorder=1)
plt.plot(x_axis, y_axis, zorder=0)
plt.show()

# Beispiel 6.7 aus dem Skript

# Datenpunkte
x = [1, 2, 3, 4]
y = [7.1, 7.9, 8.3, 8.8]


# Formel, die an die Punkte angepasst werden soll
# y = a‧ln(x + b)
def f2(x: float, coefficients: Vector) -> float:
    a, b = coefficients
    return a * math.log(x + b)


# Umformung für das gedämpfte Gauss-Newton Verfahren
# g(a,b) = y - a‧ln(x + b)
def g2(λ: Vector) -> Vector:
    return [y - f2(x, λ) for x, y in zip(x, y)]


# Lösungen der Berechnung
λ, _ = num.gauss_newton_damped(g2, [1, 1], 1e-5, 100)
a, b = λ
a = round(a, 4)
b = round(b, 4)
print(f"f2(x) = {a}·ln(x + {b})")

# Plotten des Resultates
x_axis = np.linspace(0, 5, 200)
y_axis = [f2(x, λ) for x in x_axis]

plt.scatter(x, y, color="orange", zorder=1)
plt.plot(x_axis, y_axis, zorder=0)
plt.show()

λ, count = num.gauss_newton_damped_print(g, λ_2, ε, max_iter_count, digits=5)
a, b = λ
a = round(a, 2)
b = round(b, 2)
print("λ =", [round(n, 4) for n in λ])
print(f"count = {count}")
print(f"f(x) = {a}·e^({b}·x)")
