from typing import Callable
from math import cos, sin


def int_2(a: float, b: float, f: Callable[[float], float], h: float = 1e-2) -> float:
    """
    Calculates the integral of `f` between `a` and `b` using
    [Simpson's ⅓ rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule)
    , so using polynomials of 2ⁿᵈ degree.
    """
    n = (b - a) / h

    def x(i: float) -> float:
        return a + (i * h)

    # fmt: off
    return (1/3) * h * sum(f(x(2*i - 2)) + 4*f(x(2*i - 1)) + f(x(2*i)) for i in range(1, int(n/2) + 1))
    # fmt: on


# Note: this doesn't seem to work precisely and i don't know why :sob:
def int_3(a: float, b: float, f: Callable[[float], float], h: float = 1e-2) -> float:
    """
    Calculates the integral of `f` between `a` and `b` using
    [Simpson's ⅜ rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_3/8_rule)
    , so using polynomials of 3ʳᵈ degree.
    """
    n = (b - a) / h

    def x(i: float) -> float:
        return a + (i * h)

    # fmt: off
    return (3/8) * h * sum(f(x(3*i - 3)) + 3*f(x(3*i - 2)) + 3*f(x(3*i - 1)) + f(x(3*i)) for i in range(1, int(n/3) + 1))
    # fmt: on


def int_4(a: float, b: float, f: Callable[[float], float], h: float = 1e-2) -> float:
    """
    Calculates the integral of `f` between `a` and `b` using
    [Boole's rule](https://en.wikipedia.org/wiki/Finite_difference_coefficient)
    , so using polynomials of 4ᵗʰ degree.
    """
    n = (b - a) / h

    def x(i: float) -> float:
        return a + (i * h)

    # fmt: off
    return (2/45) * h * sum(7*f(x(4*i - 4)) + 32*f(x(4*i - 3)) + 12*f(x(4*i - 2)) + 32*f(x(4*i - 1)) + 7*f(x(4*i)) for i in range(1, int(n/4) + 1))
    # fmt: on


def f(x: float) -> float:
    return cos(x)


def F(x: float) -> float:
    return sin(x)


a = 0.0
b = 5.0

print(f"Simpson's ⅓: ε = {abs(int_2(a, b, f) - (F(b) - F(a))):.2e}")
print(f"Simpson's ⅜: ε = {abs(int_3(a, b, f, 0.00001) - (F(b) - F(a))):.2e}")
print(f"Boole's :    ε = {abs(int_4(a, b, f) - (F(b) - F(a))):.2e}")
