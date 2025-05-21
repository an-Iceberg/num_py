import num
import math
from num import Vector


def f1(x: Vector) -> float:
    x1, x2, x3 = x
    return x1 + x2**2 - x3**2 - 13


def f2(x: Vector) -> float:
    x1, x2, x3 = x
    return math.log(x2 / 4) + math.exp((0.5 * x3) - 1) - 1


def f3(x: Vector) -> float:
    x1, x2, x3 = x
    return (x2 - 3) ** 2 - x3**3 + 7


def f(λ: Vector) -> Vector:
    return [
        f1(λ),
        f2(λ),
        f3(λ),
    ]


λ, count = num.newton_raphson(f, [-500, 100, 200], 10e-6, 200)
print(f"λ = {λ}")
print(f"count = {count}")
print(f"f(λ) = {[round(n, 4) for n in f(λ)]}")

num.newton_raphson_damped_print(f, [10, 10, 10], 1e-5, 100, digits=5)
