import num
import math
from num import Vector

# Beispiel Aufgabe 5.3 aus dem Skript


def f1(x: Vector) -> float:
    x1, x2, x3 = x
    return math.sin(x2 + 2 * x3)


def f2(x: Vector) -> float:
    x1, x2, x3 = x
    return math.cos(2 * x1 + x2)


def f(x: Vector) -> Vector:
    return [
        f1(x),
        f2(x),
    ]


# TODO: why does this not work?
λ, count = num.newton_raphson_print(f, [math.pi / 4, 0, math.pi], 10e-6, 200)
print(f"λ = {λ}")
print(f"count = {count}")
print(f"f(λ) = {[round(n, 4) for n in f(λ)]}")
