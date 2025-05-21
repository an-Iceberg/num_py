import num
import math
from num import Vector

# Beispiel 5.1 aus dem Skript


def f1(x: Vector) -> float:
    x1, x2 = x
    return x1**2 + x2 - 11


def f2(x: Vector) -> float:
    x1, x2 = x
    return x1 + x2**2 - 7


def f(x: Vector) -> Vector:
    return [
        f1(x),
        f2(x),
    ]


λ, count = num.newton_raphson(f, [5, 4], 10e-6, 200)
print(f"λ = {λ}")
print(f"count = {count}")
print(f"f(λ) = {[round(n, 4) for n in f(λ)]}")
