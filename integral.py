from collections.abc import Callable
from math import ceil


def int_2(a: float, b: float, f: Callable[[float], float], h: float = 1e-3) -> float:
    """
    Calculates the integral of `f` between `a` and `b` using
    [Simpson's 1/3 rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule)
    , so using polynomials of 2ⁿᵈ degree.
    """
    n = ceil((b - a) / h)

    def x(x_: float) -> float:
        return a + (x_ * h)

    def part(factor: float, subtrahend: float) -> float:
        return sum(factor * f(x(2 * i - subtrahend)) for i in range(ceil(n / 2)))
        # Σ = 0
        # for i in range(ceil(n / 2)):
        #     Σ += factor * f(x(2 * i - subtrahend))
        # return Σ

    return (1 / 3) * h * (part(1, 2) + part(4, 1) + part(1, 0))


def int_3(a: float, b: float, f: Callable[[float], float], h: float = 1e-3) -> float:
    """
    Calculates the integral of `f` between `a` and `b` using
    [Simpson's 3/8 rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_3/8_rule)
    , so using polynomials of 3ʳᵈ degree.
    """
    n = ceil((b - a) / h)

    def x(x_: float) -> float:
        return a + (x_ * h)

    n -= n % 3

    last_point = x(n)
    last_segment = int_2(last_point, b, f, h)

    def part(factor: float, subtrahend: float) -> float:
        return sum(factor * f(x(3 * i - subtrahend)) for i in range(ceil(n / 3)))
        # Σ = 0
        # for i in range(ceil(n / 3)):
        #     Σ += factor * f(x(3 * i - subtrahend))
        # return Σ

    return (3 / 8) * h * (
        part(1, 3) + part(3, 2) + part(3, 1) + part(1, 0)
    ) + last_segment


def int_4(a: float, b: float, f: Callable[[float], float], h: float = 1e-3) -> float:
    """
    Calculates the integral of `f` between `a` and `b` using
    [Boole's rule](https://en.wikipedia.org/wiki/Finite_difference_coefficient)
    , so using polynomials of 4ᵗʰ degree.
    """
    n = ceil((b - a) / h)

    def x(x_: float) -> float:
        return a + (x_ * h)

    n -= n % 4

    last_point = x(n)
    last_segment = int_2(last_point, b, f, h)

    def part(factor: float, subtrahend: float) -> float:
        return sum(factor * f(x(4 * i - subtrahend)) for i in range(ceil(n / 4)))
        # Σ = 0
        # for i in range(ceil(n / 4)):
        #     Σ += factor * f(x(4 * i - subtrahend))
        # return Σ

    return (2 / 45) * h * (
        part(7, 4) + part(32, 3) + part(12, 2) + part(32, 1) + part(7, 0)
    ) + last_segment


def int_6(a: float, b: float, f: Callable[[float], float], h: float = 1e-3) -> float:
    """
    Calculates the integral of `f` between `a` and `b` using
    [Weddle's rule](https://mathworld.wolfram.com/WeddlesRule.html)
    , so using polynomials of 4ᵗʰ degree.
    """
    n = ceil((b - a) / h)

    def x(x_: float) -> float:
        return a + (x_ * h)

    n -= n % 6

    last_point = x(n)
    last_segment = int_2(last_point, b, f, h)  # ? int_4?

    def part(factor: float, subtrahend: float) -> float:
        return sum(factor * f(x(6 * i - subtrahend)) for i in range(ceil(n / 6)))
        # Σ = 0
        # for i in range(ceil(n / 6)):
        #     Σ += factor * f(x(6 * i - subtrahend))
        # return Σ

    return (3 / 10) * h * (
        part(1, 6)
        + part(5, 5)
        + part(1, 4)
        + part(6, 3)
        + part(1, 2)
        + part(5, 1)
        + part(1, 0)
    ) + last_segment


def __int_2_old(
    a: float, b: float, f: Callable[[float], float], h: float = 1e-2
) -> float:
    """
    Calculates the integral of `f` between `a` and `b` using
    [Simpson's 1/3 rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule)
    , so using polynomials of 2ⁿᵈ degree.
    """
    n = (b - a) / h

    def x(i: float) -> float:
        return a + (i * h)

    # fmt: off
    return (1/3) * h * sum(f(x(2*i - 2)) + 4*f(x(2*i - 1)) + f(x(2*i)) for i in range(1, int(n/2) + 1))
    # fmt: on


# Note: this doesn't seem to work precisely and i don't know why :sob:
def __int_3_old(
    a: float, b: float, f: Callable[[float], float], h: float = 1e-2
) -> float:
    """
    Calculates the integral of `f` between `a` and `b` using
    [Simpson's 3/8 rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_3/8_rule)
    , so using polynomials of 3ʳᵈ degree.
    """
    n = (b - a) / h

    def x(i: float) -> float:
        return a + (i * h)

    n -= n % 3

    last_point = x(n)
    # This doesn't compute the last segment, presumably b/c it's too small
    last_segment = __int_2_old(last_point, b, f, h / 2)

    # print(f"    {last_point = }")
    # print(f"  {last_segment = }")

    # fmt: off
    return (3/8) * h * sum(f(x(3*i-3)) + 3*f(x(3*i-2)) + 3*f(x(3*i-1)) + f(x(3*i)) for i in range(1, int(n/3) + 1)) + last_segment
    # fmt: on


def __int_4_old(
    a: float, b: float, f: Callable[[float], float], h: float = 1e-2
) -> float:
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


# Note: this doesn't seem to work precisely and i don't know why :sob:
def __int_6_old(
    a: float, b: float, f: Callable[[float], float], h: float = 1e-2
) -> float:
    """
    Calculates the integral of `f` between `a` and `b` using
    [Weddle's rule](https://mathworld.wolfram.com/WeddlesRule.html)
    , so using polynomials of 4ᵗʰ degree.
    """
    n = (b - a) / h

    def x(i: float) -> float:
        return a + (i * h)

    n -= n % 6

    last_point = x(n)
    last_segment = __int_4_old(last_point, b, f, h / 2)

    # fmt: off
    return (3/10) * h * sum([f(x(6*i-6)) + 5*f(x(6*i-5)) + f(x(6*i-4)) + 6*f(x(6*i-3)) + f(x(6*i-2)) + 5*f(x(6*i-1)) + f(x(6*i)) for i in range(1, int(n/6) + 1)]) + last_segment
    # fmt: on
