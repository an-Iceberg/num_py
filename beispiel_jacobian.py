import num
import sympy as sp
import math
import numpy as np
from num import Vector

sp.init_printing()

x1, x2, x3 = sp.symbols("x1 x2 x3")

f1 = x1 + x2**2 - x3**2 - 13
f2 = sp.ln(x2 / 4) + sp.exp(0.5 * x3 - 1) - 1  # type: ignore
f3 = (x2 - 3) ** 2 - x3**3 + 7

f = sp.Matrix([f1, f2, f3])  # type: ignore
X = sp.Matrix([x1, x2, x3])
Df = f.jacobian(X)

print("Jacobian with sympy")
sp.pprint(Df.subs([(x1, 1.5), (x2, 3), (x3, 2.5)]))


def f1(input: Vector) -> float:
    x, y, z = input
    return x + y**2 - z**2 - 13


def f2(input: Vector) -> float:
    x, y, z = input
    return math.log(y / 4) + math.exp((0.5 * z) - 1) - 1


def f3(input: Vector) -> float:
    x, y, z = input
    return (y - 3) ** 2 - z**3 + 7


def f(λ: Vector) -> Vector:
    return [
        f1(λ),
        f2(λ),
        f3(λ),
    ]


x = [1.5, 3, 2.5]
ε = 0.0001

print("\nOwn Jacobian")
print(np.array_str(np.array(num.D(f, x, ε)), precision=5, suppress_small=True))
