# Author: Andràs (Sandra) Marosi

from enum import Enum
import math
from typing import List, Tuple, Callable
import numpy as np
from numpy import linalg as la
from copy import deepcopy

type Vector = List[float]
type Matrix = List[List[float]]


def d(
    f: Callable[[float], float],
    x: float,
    h: float = 0.001,
) -> float:
    """Calculates first order derivative of `f` at `x` with precision `h`."""
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)


def d2(
    f: Callable[[float], float],
    x: float,
    h: float = 0.001,
) -> float:
    """Calculates second order derivative of `f` at `x` with precision `h`."""
    return d(lambda x: d(f, x, h), x, h)
    # return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


def newton(
    f: Callable[[float], float],
    start_x: float,
    prec: float = 0.001,
    max_iter_count: int = 100,
) -> float:
    x = start_x
    x_new = x + 1
    iter_count = 0
    while abs(x_new - x) >= prec or iter_count < max_iter_count:
        iter_count += 1
        x = x_new
        x_new = x - (f(x) / d(f, x, prec))
    return x_new


def inv(x: Vector) -> Vector:
    """
    Internal helper function.\n
    Inverts a Vector element-wise.
    """
    return [-xi for xi in x]


def partial(
    f: Callable[[Vector], float],
    x: Vector,
    i: int,
    ε: float = 0.001,
) -> float:
    """
    Calculates the partial derivative (∂) of a function.

    ---
    `f` ⇒ The function of which the ∂ should be calculated. It takes a `Vector`
    as an argument, representing all the different input parameters.
    `x` ⇒ Input vector (location) at which the ∂ should be calculated.\n
    `i` ⇒ The index of the parameter. The partial is taken with respect to the
    `i`th parameter.\n
    `ε` ⇒ Precision
    """
    h = ε
    x_minus_h = deepcopy(x)
    x_plus_h = deepcopy(x)
    x_minus_h[i] -= h
    x_plus_h[i] += h
    return (f(x_plus_h) - f(x_minus_h)) / (2 * h)


def partial_vec(
    f: Callable[[Vector], Vector],
    x: Vector,
    i: int,
    ε: float = 0.001,
) -> Vector:
    """
    Calculates the partial derivative (∂) of a function.

    ---
    `f` ⇒ The function of which the ∂ should be calculated.
    It takes a `Vector` as the sole parameter.
    This list represents the different xₙ, yₙ, aₙ, bₙ and cₙ as the function parameters.\n
    `x` ⇒ Input vector (location) at which the ∂ should be calculated.\n
    `i` ⇒ The index of the parameter. The partial is taken with respect to the
    `i`th parameter.\n
    `ε` ⇒ Precision
    """
    h = ε
    x_minus_h = deepcopy(x)
    x_plus_h = deepcopy(x)
    x_minus_h[i] -= h
    x_plus_h[i] += h
    result = [a - b for a, b in zip(f(x_plus_h), f(x_minus_h))]
    result = [n / (2 * h) for n in result]
    return result


def jacobian(
    f: Callable[[Vector], Vector],
    x: Vector,
    ε: float = 0.001,
) -> Matrix:
    """
    Calculates the Jacobian matrix for `f` at `x`. The jacobian is the derivative of
    a function at a location generalised to n dimensions.

    ---
    `f` ⇒ List of functions for which the jacobian should be calculated.\n
    `x` ⇒ Location at which the jacobian is calculated.\n
    `ε` ⇒ Precision
    """
    D = []

    for i in range(len(x)):
        D.append(partial_vec(f, x, i, ε))

    # Transpose result
    return [[D[j][i] for j in range(len(D))] for i in range(len(D[0]))]


def D(f: Callable[[Vector], Vector], x: Vector, ε: float) -> Matrix:
    """
    Calculates the Jacobian matrix for `f` at `x`. The jacobian is the derivative of
    a function at a location generalised to n dimensions.

    ---
    `f` ⇒ List of functions for which the jacobian should be calculated.\n
    `x` ⇒ Location at which the jacobian is calculated.\n
    `ε` ⇒ Precision
    """
    return jacobian(f, x, ε)


def newton_raphson(
    f: Callable[[Vector], Vector], x0: Vector, ε: float, n: int
) -> Tuple[Vector, int]:
    """
    Finds a zero of `f` using the Newton-Raphson method.

    ---
    `f` ⇒ The function in question.\n
    `x0` ⇒ Starting location.\n
    `ε` ⇒ Precision\n
    `n` ⇒ Max number of iterations to do.

    ---
    ### Returns
    `Vector` ⇒ Solution\n
    `int` ⇒ Number of iterations done
    """
    x = deepcopy(x0)
    δ = [ε + 10 for _ in range(len(x0))]
    iter_count = 0

    while iter_count < n and abs(max(δ, key=abs)) >= ε:
        iter_count += 1
        δ = la.solve(D(f, x, ε), inv(f(x)))
        x += δ

    return x, iter_count


def newton_raphson_print(
    f: Callable[[Vector], Vector], x0: Vector, ε: float, n: int, digits: int = 4
) -> Tuple[Vector, int]:
    """
    Finds a zero of `f` using the Newton-Raphson method.

    ---
    `f` ⇒ The function in question.\n
    `x0` ⇒ Starting location.\n
    `ε` ⇒ Precision\n
    `n` ⇒ Max number of iterations to do.\n
    `digits` ⇒ Number of significant digits to round to when printing.

    ---
    ### Returns
    `Vector` ⇒ Solution\n
    `int` ⇒ Number of iterations done
    """
    x = deepcopy(x0)
    δ = [ε + 10 for _ in range(len(x0))]
    iter_count = 0

    print("%%%%%%%%%% NEWTON-RAPHSON %%%%%%%%%%\n")

    while iter_count < n and abs(max(δ, key=abs)) >= ε:
        Df = D(f, x, ε)
        fx_neg = inv(f(x))
        δ = la.solve(Df, fx_neg)
        x += δ
        iter_count += 1

        print(f"===== Iteration {iter_count}: ==========")
        print("Df(x) =")
        print(
            "  "
            + np.array_str(np.array(Df), precision=digits, suppress_small=True).replace(
                "\n", "\n  "
            )
        )
        print(f"-f(x) = {[round(n, digits) for n in fx_neg]}")
        print(f"δ = {np.array_str(δ, precision=digits, suppress_small=True)}")
        print(f"x = {[round(n, digits) for n in x]}")
        print("=============================")
        print()

    print("%%%%%%%%%% DONE %%%%%%%%%%")

    return x, iter_count


def newton_raphson_damped(
    f: Callable[[Vector], Vector], x0: Vector, ε: float, n: int
) -> Tuple[Vector, int]:
    """
    Finds a zero of `f` using the damped Newton-Raphson method for an `f`.

    ---
    `f` ⇒ The function in question.\n
    `x0` ⇒ Starting location.\n
    `ε` ⇒ Precision\n
    `n` ⇒ Max number of iterations to do.

    ---
    ### Returns
    `Vector` ⇒ Solution\n
    `int` ⇒ Number of iterations done
    """
    x = deepcopy(x0)
    δ = [ε + 10 for _ in range(len(x0))]
    iter_count = 0

    while iter_count < n and abs(max(δ, key=abs)) >= ε:
        iter_count += 1
        δ = la.solve(D(f, x, ε), inv(f(x)))
        k = 0
        while la.norm(f(x + (δ / 2**k)), 2) > la.norm(f(x), 2):
            k += 1
        x += δ / 2**k

    return x, iter_count


def newton_raphson_damped_print(
    f: Callable[[Vector], Vector], x0: Vector, ε: float, n: int, digits: int = 4
) -> Tuple[Vector, int]:
    """
    Finds a zero of `f` using the damped Newton-Raphson method for an `f`.

    ---
    `f` ⇒ The function in question.\n
    `x0` ⇒ Starting location.\n
    `ε` ⇒ Precision\n
    `n` ⇒ Max number of iterations to do.\n
    `digits` ⇒ Number of significant digits to round to when printing.

    ---
    ### Returns
    `Vector` ⇒ Solution\n
    `int` ⇒ Number of iterations done
    """
    x = deepcopy(x0)
    δ = [ε + 10 for _ in range(len(x0))]
    iter_count = 0

    print("%%%%%%%%%% NEWTON-RAPHSON DAMPED %%%%%%%%%%\n")

    while iter_count < n and abs(max(δ, key=abs)) >= ε:
        iter_count += 1

        Df = D(f, x, ε)
        fx_neg = inv(f(x))
        δ = la.solve(Df, fx_neg)

        print(f"===== Iteration {iter_count}: ==========")
        print("Df(x) =")
        print(
            "  "
            + np.array_str(np.array(Df), precision=digits, suppress_small=True).replace(
                "\n", "\n  "
            )
        )
        print(f"-f(x) = {[round(n, digits) for n in fx_neg]}")
        print(f"δ = {np.array_str(δ, precision=digits, suppress_small=True)}")

        k = 0
        norm_damped = la.norm(f(x + (δ / 2**k)), 2)  # type: ignore
        norm_undamped = la.norm(f(x), 2)
        while norm_damped > norm_undamped:
            print(f"  k = {k}")
            print(f"    ||f(λ + δ/2**k)||₂ = {round(norm_damped, digits)}")
            print(f"    ||f(λ)||₂ = {round(norm_undamped, digits)}")
            k += 1
            norm_damped = la.norm(f(x + (δ / 2**k)), 2)
            norm_undamped = la.norm(f(x), 2)

        x += δ / 2**k

        print(f"k = {k}")
        print(f"  ||f(λ + δ/2**k)||₂ = {round(norm_damped, digits)}")
        print(f"  ||f(λ)||₂ = {round(norm_undamped, digits)}")
        print(f"x = {[round(n, digits) for n in x]}")
        print("=============================")
        print()

    print("%%%%%%%%%% DONE %%%%%%%%%%")

    return x, iter_count


# %%%%%%%%%%%%%%%%%%%%%%% INTERPOLATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def __lagrange(
    x: float, n: int, i: int, x_vec: Vector, printing=False, digits: int = 4
) -> float:
    """
    ##### Author: Raphaël Monteiro
    """
    Π = 1
    if printing:
        print(f"l{i}(x = {round(x, digits)}) =")
    for j in range(n):
        if j != i:
            Π *= (x - x_vec[j]) / (x_vec[i] - x_vec[j])
            if printing:
                print(
                    f"  · ({round(x, digits)}-{round(x_vec[j], digits)})÷({round(x_vec[i], digits)}-{round(x_vec[j], digits)})",
                )
    if printing:
        print(f"  = {round(Π, digits)}")
    return Π


def lagrange_point(
    x_point: float, x: Vector, y: Vector, printing=False, digits: int = 4
) -> float:
    """
    ##### Author: Raphaël Monteiro

    Calculates `y` of `x_point` given `x` and `y` vectors of data values.

    ---
    `x_point` ⇒ The point for which `l(x)` should be calculated.\n
    `x` ⇒ Data points on the x-axis.\n
    `y` ⇒ Data points on the y-axis.\n
    `printing` ⇒ If `True`, prints intermediate results into the console.

    ---
    ### Returns:
    `float` ⇒ The calculated y value at `x`.
    """
    if len(x) != len(y):
        raise Exception("x and y vectors need to have the same length!")
    n = len(x)
    Σ = 0
    ls = []
    for i in range(n):
        l_point = __lagrange(x_point, n, i, x, printing, digits)
        Σ += l_point * y[i]
        ls.append(l_point)
    if printing:
        print(f"P{n}(x) = ", end="")
        for i in range(n):
            print(f"l{i}(x)·y{i}", end="")
            if i != n - 1:
                print(" + ", end="")
        print(" =")

        print(f"P{n}({round(x_point, digits)}) = ", end="")
        for i in range(n):
            print(f"{round(ls[i], digits)}·{round(y[i], digits)}", end="")
            if i != n - 1:
                print(" + ", end="")
        print(f" = {round(Σ, digits)}")
    return Σ


def lagrange_vec(x: Vector, x_data: Vector, y_data: Vector) -> Vector:
    """
    ##### Author: Raphaël Monteiro

    Calculates the `y` values for all given `x` values using lagrange polynomial interpolation.

    ---
    `x` ⇒ Vector containing all values for which the polynomial should be evaluated.\n
    `x_data` ⇒ Data points on the x axis.\n
    `x_data` ⇒ Data points on the y axis.\n

    ---
    ### Returns
    `Vector` ⇒ The calculated `y` values, corresponding to the given `x` values.
    """
    if len(x_data) != len(y_data):
        raise Exception("x and y vectors need to have the same length!")
    y = list(np.zeros(len(x)))
    for i in range(0, len(x)):
        y[i] = lagrange_point(x[i], x_data, y_data)
    return y


def spline_print(x: Vector, x_data: Vector, y_data: Vector, digits: int = 4) -> Vector:
    """
    Interpolates between given data points using cubic spline interpolation (natural splines).

    ---
    `x` ⇒ Vector of `x`s for which the `y`s should be calculated.\n
    `x_data` ⇒ Data points on the x axis.\n
    `x_data` ⇒ Data points on the y axis.\n
    `digits` ⇒ Number of significant digits to round to when printing.

    ---
    ### Returns
    `Vector` ⇒ The calculated `y` values.
    """
    if len(x_data) != len(y_data):
        raise Exception("x & y must be of same length")

    n = len(x_data)

    n -= 1

    A = [[0.0 for _ in range(n)] for _ in range(n)]
    h = [0.0 for _ in range(n)]
    a = [0.0 for _ in range(n)]
    b = [0.0 for _ in range(n)]
    c = [0.0 for _ in range(n)]
    d = [0.0 for _ in range(n)]
    z = [0.0 for _ in range(n)]

    for i in range(n):
        # 1
        a[i] = y_data[i]
        print(f"a{i} = y{i} = {a[i]}")
        # 2
        h[i] = x_data[i + 1] - x_data[i]
        print(
            f"h{i} = x{i + 1}-x{i} = {round(x_data[i + 1], digits)}-{round(x_data[i], digits)} = {round(h[i], digits)}"
        )

    print(f"\nh = {np.array_str(np.array(h), precision=digits, suppress_small=True)}\n")

    # 4 a
    A[0][0] = 2 * (h[0] + h[1])
    print(
        f"A0,0 = 2·(h0+h1) = 2·({round(h[0], digits)}+{round(h[1], digits)}) = {round(A[0][0], digits)}"
    )
    # 4 b
    for i in range(1, n - 1):
        A[i][i] = 2 * (h[i] + h[i + 1])
        print(
            f"A{i},{i} = 2·(h{i}+h{i + 1}) = 2·({round(h[i], digits)}+{round(h[i + 1], digits)}) = {round(A[i][i], digits)}"
        )
        A[i + 1][i] = h[i + 1]
        print(f"A{i + 1},{i} = h{i + 1} = {round(A[i + 1][i], digits)}")
        A[i][i + 1] = h[i + 1]
        print(f"A{i},{i + 1} = h{i + 1} = {round(A[i][i + 1], digits)}")
    # 4 c
    A[-1][-1] = 2 * (h[-2] + h[-1])
    print(
        f"A{len(A) - 1},{len(A[0]) - 1} = 2·(h{len(h) - 2}+h{len(h) - 1}) = 2·({round(h[0], digits)}+{round(h[1], digits)}) = {round(A[0][0], digits)}"
    )

    print("\nA =")
    print(
        "  "
        + np.array_str(np.array(A), precision=digits, suppress_small=True).replace(
            "\n", "\n  "
        )
    )
    print()

    # 4 a
    z[0] = 3 * ((y_data[2] - y_data[1]) / h[1]) - 3 * ((y_data[1] - y_data[0]) / h[0])
    print(
        f"z0 = 3·(y2-y1 ÷ h1) - 3·(y1-y0 ÷ h0) = 3·({round(y_data[2], digits)}-{round(y_data[2], digits)} ÷ {round(h[1], digits)}) - 3·({round(y_data[1], digits)}-{round(y_data[0], digits)} ÷ {round(h[0], digits)}) = {round(z[0], digits)}"
    )
    for i in range(1, n - 1):
        # 4 b
        z[i] = 3 * ((y_data[i + 1] - y_data[i]) / h[i]) - 3 * (
            (y_data[i] - y_data[i - 1]) / h[i - 1]
        )
        print(
            f"z{i} = 3·(y{i + 1}-y{i} ÷ h{i}) - 3·(y{i}-y{i - 1} ÷ h{i - 1}) = 3·({round(y_data[i + 1], digits)}-{round(y_data[i], digits)} ÷ {round(h[i], digits)}) - 3·({round(y_data[i], digits)}-{round(y_data[i - 1], digits)} ÷ {round(h[i - 1], digits)}) = {round(z[i], digits)}"
        )
    # 4 c
    z[-1] = 3 * ((y_data[-1] - y_data[-2]) / h[-2]) - 3 * (
        (y_data[-2] - y_data[-3]) / h[-3]
    )
    print(
        f"z{len(z) - 1} = 3·(y{len(y_data) - 1}-y{len(y_data) - 2} ÷ h{len(h) - 2}) - 3·(y{len(y_data) - 2}-y{len(y_data) - 3} ÷ h{len(h) - 3}) = 3·({round(y_data[-1], digits)}-{round(y_data[-2], digits)} ÷ {round(h[-2], digits)}) - 3·({round(y_data[-2], digits)}-{round(y_data[-3], digits)} ÷ {round(h[-3], digits)}) = {round(z[-1], digits)}"
    )

    print(f"\nz = {np.array_str(np.array(z), precision=digits, suppress_small=True)}\n")

    c = la.solve(A, z)

    # 3
    c[0] = 0
    # c[-1] = 0

    print("c = np.linalg.solve(A, z) = ", end="")
    print(f"{[round(c, digits) for c in c]}", end="\n\n")

    for i in range(n - 1):
        # 5
        b[i] = ((y_data[i + 1] - y_data[i]) / h[i]) - (h[i] / 3) * (
            c[i + 1] + (2 * c[i])
        )
        print(
            f"b{i} = (y{i + 1}-y{i} ÷ h{i}) - (h{i}÷3)·(c{i + 1} + 2·c{i}) = ({round(y_data[i + 1], digits)}-{round(y_data[i], digits)} ÷ {round(h[i], digits)}) - ({round(h[i], digits)}÷3)·({round(c[i + 1], digits)} + 2·{round(c[i], digits)}) = {round(b[i], digits)}"
        )
        # 6
        d[i] = 1 / (3 * h[i]) * (c[i + 1] - c[i])
        print(
            f"d{i} = 1÷(3·h{i})·(c{i + 1}-c{i}) = 1÷(3·{round(h[i], digits)})·({round(c[i + 1], digits)} - {round(c[i], digits)}) = {round(d[i], digits)}",
            end="\n",
        )
    # 5
    b[-1] = ((y_data[-1] - y_data[-2]) / h[-1]) - (h[-1] / 3) * 2 * c[-1]
    print(
        f"b{len(b) - 1} = (y{len(y_data) - 1}-y{len(y_data) - 2} ÷ h{len(h) - 1}) - (h{len(h) - 1}÷3)·2·c{len(c) - 1} = ({round(y_data[-1], digits)}-{round(y_data[-2], digits)} ÷ {round(h[-1], digits)}) - ({round(h[-1], digits)}÷3)·2·{round(c[-1], digits)} = {round(b[-1], digits)}"
    )
    # 6
    d[-1] = 1 / (3 * h[-1]) * (-1) * c[-1]
    print(
        f"d{len(d) - 1} = 1÷(3·h{len(h) - 1})·-c{len(c) - 1} = 1÷(3·{round(h[-1], digits)})·-{round(c[-1], digits)} = {round(d[-1], digits)}",
        end="\n\n",
    )

    print(f"a = {np.array_str(np.array(a), precision=digits, suppress_small=True)}")
    print(f"b = {np.array_str(np.array(b), precision=digits, suppress_small=True)}")
    print(f"c = {np.array_str(np.array(c), precision=digits, suppress_small=True)}")
    print(f"d = {np.array_str(np.array(d), precision=digits, suppress_small=True)}")

    y = [0.0 for _ in range(len(x))]

    for i in range(len(x)):
        for j in range(n):
            # i = index of x
            # j = index of interval

            # First interval
            if x[i] < x_data[0]:
                y[i] = (
                    a[0]
                    + b[0] * (x[i] - x_data[0])
                    + c[0] * (x[i] - x_data[0]) ** 2
                    + d[0] * (x[i] - x_data[0]) ** 3
                )
            # Between intervals
            elif x_data[j] <= x[i] and x[i] <= x_data[j + 1]:
                y[i] = (
                    a[j]
                    + b[j] * (x[i] - x_data[j])
                    + c[j] * (x[i] - x_data[j]) ** 2
                    + d[j] * (x[i] - x_data[j]) ** 3
                )
            # Last interval
            elif x_data[-1] < x[i]:
                y[i] = (
                    a[-1]
                    + b[-1] * (x[i] - x_data[-2])
                    + c[-1] * (x[i] - x_data[-2]) ** 2
                    + d[-1] * (x[i] - x_data[-2]) ** 3
                )

    return y


def spline(x: Vector, x_data: Vector, y_data: Vector) -> Vector:
    """
    Interpolates between given data points using cubic spline interpolation (natural splines).

    ---
    `x` ⇒ Vector of `x`s for which the `y`s should be calculated.\n
    `x_data` ⇒ Data points on the x axis.\n
    `x_data` ⇒ Data points on the y axis.\n
    `digits` ⇒ Number of significant digits to round to when printing.

    ---
    ### Returns
    `Vector` ⇒ The calculated `y` values.
    """
    if len(x_data) != len(y_data):
        raise Exception("x & y must be of same length")

    n = len(x_data)

    n -= 1

    A = [[0.0 for _ in range(n)] for _ in range(n)]
    h = [0.0 for _ in range(n)]
    a = [0.0 for _ in range(n)]
    b = [0.0 for _ in range(n)]
    c = [0.0 for _ in range(n)]
    d = [0.0 for _ in range(n)]
    z = [0.0 for _ in range(n)]

    for i in range(n):
        # 1
        a[i] = y_data[i]
        # 2
        h[i] = x_data[i + 1] - x_data[i]

    # 4 a
    A[0][0] = 2 * (h[0] + h[1])
    # 4 b
    for i in range(1, n - 1):
        A[i][i] = 2 * (h[i] + h[i + 1])
        A[i + 1][i] = h[i + 1]
        A[i][i + 1] = h[i + 1]
    # 4 c
    A[-1][-1] = 2 * (h[-2] + h[-1])

    # 4 a
    z[0] = 3 * ((y_data[2] - y_data[1]) / h[1]) - 3 * ((y_data[1] - y_data[0]) / h[0])
    for i in range(1, n - 1):
        # 4 b
        z[i] = 3 * ((y_data[i + 1] - y_data[i]) / h[i]) - 3 * (
            (y_data[i] - y_data[i - 1]) / h[i - 1]
        )
    # 4 c
    z[-1] = 3 * ((y_data[-1] - y_data[-2]) / h[-2]) - 3 * (
        (y_data[-2] - y_data[-3]) / h[-3]
    )

    c = la.solve(A, z)

    # 3
    c[0] = 0
    # c[-1] = 0

    for i in range(n - 1):
        # 5
        b[i] = ((y_data[i + 1] - y_data[i]) / h[i]) - (h[i] / 3) * (
            c[i + 1] + (2 * c[i])
        )
        # 6
        d[i] = 1 / (3 * h[i]) * (c[i + 1] - c[i])
    # 5
    b[-1] = ((y_data[-1] - y_data[-2]) / h[-1]) - (h[-1] / 3) * 2 * c[-1]
    # 6
    d[-1] = 1 / (3 * h[-1]) * (-1) * c[-1]

    y = [0.0 for _ in range(len(x))]

    for i in range(len(x)):
        for j in range(n):
            # i = index of x
            # j = index of interval

            # First interval
            if x[i] < x_data[0]:
                y[i] = (
                    a[0]
                    + b[0] * (x[i] - x_data[0])
                    + c[0] * (x[i] - x_data[0]) ** 2
                    + d[0] * (x[i] - x_data[0]) ** 3
                )
            # Between intervals
            elif x_data[j] <= x[i] and x[i] <= x_data[j + 1]:
                y[i] = (
                    a[j]
                    + b[j] * (x[i] - x_data[j])
                    + c[j] * (x[i] - x_data[j]) ** 2
                    + d[j] * (x[i] - x_data[j]) ** 3
                )
            # Last interval
            elif x_data[-1] < x[i]:
                y[i] = (
                    a[-1]
                    + b[-1] * (x[i] - x_data[-2])
                    + c[-1] * (x[i] - x_data[-2]) ** 2
                    + d[-1] * (x[i] - x_data[-2]) ** 3
                )

    return y


# %%%%%%%%%%%%%%%%%%%%%%% AUSGLEICHSRECHNUGNEN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def gauss_newton(
    g: Callable[[Vector], Vector], λ0: Vector, ε: float, n: int
) -> Tuple[Vector, int]:
    """
    Finds the nearest local minimum of `f` near `λ₀` using the Gauss-Newton method.

    g(λ) := y - f(λ)

    ---
    `g` ⇒ The function in question.\n
    `λ₀` ⇒ Starting vector.\n
    `ε` ⇒ Precision\n
    `n` ⇒ Max number of iterations to do.

    ---
    ### Returns
    `Vector` ⇒ Solution\n
    `int` ⇒ Number of iterations done
    """
    iter_count = 0
    λ = deepcopy(λ0)
    increment = ε + 1

    while iter_count < n and increment > ε:
        iter_count += 1
        Q, R = la.qr(D(g, λ, ε))
        δ = la.solve(R, -Q.T @ g(λ)).flatten()
        λ += δ
        increment = la.norm(δ, 2)

    return list(λ), iter_count


def gauss_newton_print(
    g: Callable[[Vector], Vector], λ0: Vector, ε: float, n: int, digits: int = 4
) -> Tuple[Vector, int]:
    """
    Finds the nearest local minimum of `f` near `λ₀` using the Gauss-Newton method.

    g(λ) := y - f(λ)

    ---
    `g` ⇒ The function in question.\n
    `λ₀` ⇒ Starting vector.\n
    `ε` ⇒ Precision\n
    `n` ⇒ Max number of iterations to do.\n
    `digits` ⇒ Number of significant digits to round to when printing.

    ---
    ### Returns
    `Vector` ⇒ Solution\n
    `int` ⇒ Number of iterations done
    """
    iter_count = 0
    λ = deepcopy(λ0)
    increment = ε + 1
    error_functional = la.norm(g(λ)) ** 2

    # np.set_printoptions(precision=digits, suppress=True)

    print("%%%%%%%%%%%%%%%%%%%% GAUSS-NEWTON %%%%%%%%%%%%%%%%%%%%", end="\n\n")

    while iter_count < n and increment > ε:
        Dg = D(g, λ, ε)
        gλ = g(λ)
        Q, R = la.qr(Dg)
        δ = la.solve(R, -Q.T @ gλ).flatten()
        λ += δ
        error_functional = la.norm(gλ) ** 2
        increment = la.norm(δ, 2)
        iter_count += 1

        print("========= Iteration {} =============".format(iter_count))
        print("Dg(λ) =")
        print(
            "  "
            + np.array_str(np.array(Dg), precision=digits, suppress_small=True).replace(
                "\n", "\n  "
            )
        )
        print("g(λ) = {}".format([round(n, digits) for n in gλ]))
        print("Q =")
        print("  " + str(Q).replace("\n", "\n  "))
        print("R =")
        print("  " + str(R).replace("\n", "\n  "))
        print(f"δ = {np.array_str(δ, precision=digits, suppress_small=True)}")
        print(f"λ = {[round(n, digits) for n in λ]}")
        print(f"Ẽ(λ) = {round(error_functional, digits)}")
        print(f"increment = {round(increment, digits)}")
        print("====================================")
        print()

    print("%%%%%%%%%%%%%%%%%%%% DONE %%%%%%%%%%%%%%%%%%%%", end="\n\n")

    return list(λ), iter_count


def gauss_newton_damped(
    g: Callable[[Vector], Vector], λ0: Vector, ε: float, n: int
) -> Tuple[Vector, int]:
    """
    Finds the nearest local minimum of `f` near `λ₀` using the Gauss-Newton method with damping.

    g(λ) := y - f(λ)

    ---
    `g` ⇒ The function in question.\n
    `λ₀` ⇒ Starting vector.\n
    `ε` ⇒ Precision\n
    `n` ⇒ Max number of iterations to do.\n

    ---
    ### Returns
    `Vector` ⇒ Solution\n
    `int` ⇒ Number of iterations done
    """
    iter_count = 0
    λ = deepcopy(λ0)
    inc = ε + 1

    while iter_count < n and inc > ε:
        iter_count += 1
        Q, R = la.qr(D(g, λ, ε))
        gλ = g(λ)
        δ = la.solve(R, -Q.T @ gλ).flatten()
        p = 0

        while la.norm(g(λ + (δ / 2**p)), 2) ** 2 > la.norm(gλ, 2) ** 2:
            p += 1

        λ += δ / 2**p
        inc = la.norm(δ, 2)

    return list(λ), iter_count


def gauss_newton_damped_print(
    g: Callable[[Vector], Vector], λ0: Vector, ε: float, n: int, digits: int = 4
) -> Tuple[Vector, int]:
    """
    Finds the nearest local minimum of `f` near `λ₀` using the Gauss-Newton method with damping.

    ---
    `g` ⇒ The function in question.\n
    `λ₀` ⇒ Starting vector.\n
    `ε` ⇒ Precision\n
    `n` ⇒ Max number of iterations to do.\n
    `digits` ⇒ Number of significant digits to round to when printing.

    ---
    ### Returns
    `Vector` ⇒ Solution\n
    `int` ⇒ Number of iterations done
    """
    iter_count = 0
    λ = deepcopy(λ0)
    inc = ε + 1

    print("%%%%%%%%%%%%%%%%%%%% GAUSS-NEWTON DAMPED %%%%%%%%%%%%%%%%%%%%", end="\n\n")

    while iter_count < n and inc > ε:
        iter_count += 1
        Dg = D(g, λ, ε)
        gλ = g(λ)
        Q, R = la.qr(Dg)
        δ = la.solve(R, -Q.T @ gλ).flatten()

        print("========= Iteration {} =============".format(iter_count))
        print("Dg(λ) =")
        print(
            "  "
            + np.array_str(np.array(Dg), precision=digits, suppress_small=True).replace(
                "\n", "\n  "
            )
        )
        print("g(λ) = {}".format([round(n, digits) for n in gλ]))
        print("Q =")
        print("  " + str(Q).replace("\n", "\n  "))
        print("R =")
        print("  " + str(R).replace("\n", "\n  "))
        print(f"δ = {δ}")

        p = 0

        norm_damped = la.norm(g(λ + (δ / 2**p)), 2) ** 2  # type: ignore
        norm_undamped = la.norm(gλ, 2) ** 2
        while norm_damped > norm_undamped:
            print(f"  p = {p}")
            print(f"    ||g(λ + δ/2**p)||₂² = {round(norm_damped, digits)}")
            print(f"    ||g(λ)||₂² = {round(norm_undamped, digits)}")
            p += 1
            norm_damped = la.norm(g(λ + (δ / 2**p)), 2) ** 2
            norm_undamped = la.norm(gλ, 2) ** 2

        λ += δ / 2**p
        inc = la.norm(δ, 2)
        error_functional = la.norm(gλ) ** 2

        print(f"p = {p}")
        print(f"  ||g(λ + δ/2**p)||₂² = {round(norm_damped, digits)}")
        print(f"  ||g(λ)||₂² = {round(norm_undamped, digits)}")
        print(
            f"δ ÷ 2**p = {np.array_str((δ / 2**p), precision=digits, suppress_small=True)}"
        )
        print(f"λ = {[round(n, digits) for n in λ]}")
        print(f"Ẽ(λ) = {round(error_functional, digits)}")
        print(f"increment = {round(inc, digits)}")
        print("====================================")
        print()

    print("%%%%%%%%%%%%%%%%%%%% DONE %%%%%%%%%%%%%%%%%%%%", end="\n\n")

    return list(λ), iter_count


# %%%%%%%%%%%%%%%%%%%%%%% INTEGRATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def rect(f: Callable[[float], float], a: float, b: float, h: float) -> float:
    """
    Calculates the integral of `f` in the interval between `a` and `b` using the
    rectangles method.

    ---
    `f` : The function to be integrated\n
    `a`: Lower bound\n
    `b`: Upper bound\n
    `h`: The width of each segment
    """
    n = int(math.ceil((b - a) / h))
    Σ = 0
    for i in range(n):
        Σ += f(a + (i * h) + (h / 2))
    return h * Σ


def rect_print(
    f: Callable[[float], float], a: float, b: float, h: float, digits: int = 4
) -> float:
    """
    Calculates the integral of `f` in the interval between `a` and `b` using the
    rectangles method and prints each intermediate result into the console.

    ---
    `f` : The function to be integrated\n
    `a`: Lower bound\n
    `b`: Upper bound\n
    `h`: The width of each segment\n
    `digits` ⇒ Number of significant digits to round to when printing.
    """
    n = int(math.ceil((b - a) / h))
    Σ = 0
    print(f"Rf({round(h, digits)}):")
    for i in range(n):
        σ = f(a + (i * h) + (h / 2))
        Σ += σ
        print(
            f" + f(a + {i}·h + ½h)= f({a} + {i}·{round(h, digits)} + {round(h / 2, digits)}) = f({round(a + (i * h) + (h / 2), digits)}) = {round(σ, digits)}"
        )
    print(f" Σ = {round(Σ, digits)}")
    print("Rf(h) = h·Σ")
    print(f"  = {round(h, digits)}·{round(Σ, digits)}")
    print(f"  = {round(h * Σ, digits)}")
    return h * Σ


def R(f: Callable[[float], float], a: float, b: float, h: float) -> float:
    """
    Calculates the integral of `f` in the interval between `a` and `b` using the
    rectangles method.

    ---
    `f` : The function to be integrated\n
    `a`: Lower bound\n
    `b`: Upper bound\n
    `h`: The width of each segment
    """
    return rect(f, a, b, h)


def R_print(
    f: Callable[[float], float], a: float, b: float, h: float, digits: int = 4
) -> float:
    """
    Calculates the integral of `f` in the interval between `a` and `b` using the
    rectangles method and prints each intermediate result into the console.

    ---
    `f` : The function to be integrated\n
    `a`: Lower bound\n
    `b`: Upper bound\n
    `h`: The width of each segment\n
    `digits` ⇒ Number of significant digits to round to when printing.
    """
    return rect_print(f, a, b, h, digits)


def trapez(f: Callable[[float], float], a: float, b: float, h: float) -> float:
    """
    Calculates the integral of `f` in the interval between `a` and `b` using the
    trapezoid method.

    ---
    `f` : The function to be integrated\n
    `a`: Lower bound\n
    `b`: Upper bound\n
    `h`: The width of each segment
    """
    n = int(math.ceil((b - a) / h))
    Σ = 0
    for i in range(1, n):
        Σ += f(a + (i * h))
    return h * (((f(a) + f(b)) / 2) + Σ)


def trapez_print(
    f: Callable[[float], float], a: float, b: float, h: float, digits: int = 4
) -> float:
    """
    Calculates the integral of `f` in the interval between `a` and `b` using the
    trapezoid method and prints each intermediate result into the console.

    ---
    `f` : The function to be integrated\n
    `a`: Lower bound\n
    `b`: Upper bound\n
    `h`: The width of each segment\n
    `digits` ⇒ Number of significant digits to round to when printing.
    """
    n = int(math.ceil((b - a) / h))
    Σ = 0
    print(f"Tf({round(h, digits)}):")
    for i in range(1, n):
        σ = f(a + (i * h))
        Σ += σ
        print(
            f" + f(a + {i}·h) = f({round(a, digits)} + {i}·{round(h, digits)}) = f({round(a + (i * h), digits)}) = {round(σ, digits)}"
        )
    print(f" Σ = {round(Σ, digits)}")
    print("Tf(h) = h·(½(f(a) + f(b)) + Σ)")
    print(
        f"  = {round(h, digits)}·(½({round(f(a), digits)} + {round(f(b), digits)}) + {round(Σ, digits)})"
    )
    print(f"  = {round(h * (((f(a) + f(b)) / 2) + Σ), digits)}")
    return h * (((f(a) + f(b)) / 2) + Σ)


def T(f: Callable[[float], float], a: float, b: float, h: float) -> float:
    """
    Calculates the integral of `f` in the interval between `a` and `b` using the
    trapezoid method.

    ---
    `f` : The function to be integrated\n
    `a`: Lower bound\n
    `b`: Upper bound\n
    `h`: The width of each segment
    """
    return trapez(f, a, b, h)


def T_print(
    f: Callable[[float], float], a: float, b: float, h: float, digits: int = 4
) -> float:
    """
    Calculates the integral of `f` in the interval between `a` and `b` using the
    trapezoid method and prints each intermediate result into the console.

    ---
    `f` : The function to be integrated\n
    `a`: Lower bound\n
    `b`: Upper bound\n
    `h`: The width of each segment\n
    `digits` ⇒ Number of significant digits to round to when printing.
    """
    return trapez_print(f, a, b, h, digits)


def simpson(f: Callable[[float], float], a: float, b: float, h: float) -> float:
    """
    Calculates the integral of `f` in the interval between `a` and `b` using Simpson's rule.

    ---
    `f` : The function to be integrated\n
    `a`: Lower bound\n
    `b`: Upper bound\n
    `h`: The width of each segment
    """
    n = int(math.ceil((b - a) / h))
    Σ1 = 0
    Σ2 = 0
    for i in range(1, n):
        Σ1 += f(a + (i * h))
    for i in range(1, n + 1):
        Σ2 += f(((a + ((i - 1) * h)) + (a + (i * h))) / 2)
    return (h / 3) * ((f(a) / 2) + Σ1 + (2 * Σ2) + (f(b) / 2))


def simpson_print(
    f: Callable[[float], float], a: float, b: float, h: float, digits: int = 4
) -> float:
    """
    Calculates the integral of `f` in the interval between `a` and `b` using
    Simpson's rule and prints each intermediate result into the console.

    ---
    `f` : The function to be integrated\n
    `a`: Lower bound\n
    `b`: Upper bound\n
    `h`: The width of each segment\n
    `digits` ⇒ Number of significant digits to round to when printing.
    """
    n = int(math.ceil((b - a) / h))
    Σ1 = 0
    Σ2 = 0
    print(f"Sf({round(h, digits)}):")
    for i in range(1, n):
        σ1 = f(a + (i * h))
        Σ1 += σ1
        print(
            f" + f(a + {i}·h) = f({round(a, digits)} + {i}·{round(h, digits)}) = f({round(a + (i * h), digits)}) = {round(σ1, digits)}"
        )
    print(f" Σ₁ = {round(Σ1, digits)}")
    for i in range(1, n + 1):
        σ2 = f(((a + ((i - 1) * h)) + (a + (i * h))) / 2)
        Σ2 += σ2
        print(
            f" + f(½(a + h·{i - 1} + a + {i}·h)) = f(½({round(a, digits)} + {round(i - 1, digits)}·{round(h, digits)} + {round(a, digits)} + {round(i, digits)}·{round(h, digits)})) = f({round(((a + ((i - 1) * h)) + (a + (i * h))) / 2, digits)}) = {round(σ2, digits)}"
        )
    print(f" Σ₂ = {round(Σ2, digits)}")
    result = (h / 3) * ((f(a) / 2) + Σ1 + (2 * Σ2) + (f(b) / 2))
    print("Sf(h) = ⅓h·(½f(a) + Σ₁ + 2·Σ₂ + ½f(b))")
    print(
        f"  = {round(h / 3, digits)}·(½·{round(f(a), digits)} + {round(Σ1, digits)} + 2·{round(Σ2, digits)} + ½·{round(f(b), digits)})"
    )
    print(f"  = {round(result, digits)}")
    return result


def S(f: Callable[[float], float], a: float, b: float, h: float) -> float:
    """
    Calculates the integral of `f` in the interval between `a` and `b` using Simpson's rule.

    ---
    `f` : The function to be integrated\n
    `a`: Lower bound\n
    `b`: Upper bound\n
    `h`: The width of each segment
    """
    return simpson(f, a, b, h)


def S_print(
    f: Callable[[float], float], a: float, b: float, h: float, digits: int = 4
) -> float:
    """
    Calculates the integral of `f` in the interval between `a` and `b` using
    Simpson's rule and prints each intermediate result into the console.

    ---
    `f` : The function to be integrated\n
    `a`: Lower bound\n
    `b`: Upper bound\n
    `h`: The width of each segment\n
    `digits` ⇒ Number of significant digits to round to when printing.
    """
    return simpson_print(f, a, b, h, digits)


class IntMethod(Enum):
    """
    Chose from:
    - Trapez method
    - Rectangle method
    - Simpson's rule
    """

    rect = 1
    trapez = 2
    simpson = 3


def romberg(
    f: Callable[[float], float],
    a: float,
    b: float,
    depth: int,
    method: IntMethod,
) -> float:
    """
    Calculates the integral of `f` between `a` and `b` using Romberg-extrapolation.

    ---
    `f` ⇒ Function to be integrated\n
    `a` ⇒ Lower bound\n
    `b` ⇒ Upper bound\n
    `depth` ⇒ Depth of recursion tree\n
    `method` ⇒ One of the three integration methods provided in this module:
    - `rect`
    - `trapez`
    - `simpson`
    """
    return __romberg_jk(f, a, b, 0, depth, method)


def romberg_print(
    f: Callable[[float], float],
    a: float,
    b: float,
    depth: int,
    method: IntMethod,
    digits: int = 4,
    print_int_results=False,
) -> float:
    """
    Calculates the integral of `f` between `a` and `b` using Romberg-extrapolation.

    ---
    `f` ⇒ Function to be integrated\n
    `a` ⇒ Lower bound\n
    `b` ⇒ Upper bound\n
    `depth` ⇒ Depth of recursion tree\n
    `method` ⇒ One of the three integration methods provided in this crate:
    - `rect`
    - `trapez`
    - `simpson`\n
    `digits` ⇒ Number of significant digits to round to when printing.
    """
    return __romberg_jk_print(f, a, b, 0, depth, method, digits, print_int_results)


def __romberg_jk(
    f: Callable[[float], float],
    a: float,
    b: float,
    j: int,
    k: int,
    method: IntMethod,
) -> float:
    if k == 0:
        match method:
            case IntMethod.rect:
                return R(f, a, b, (b - a) / 2**j)
            case IntMethod.trapez:
                return T(f, a, b, (b - a) / 2**j)
            case IntMethod.simpson:
                return S(f, a, b, (b - a) / 2**j)
    return (
        4**k * __romberg_jk(f, a, b, j + 1, k - 1, method)
        - __romberg_jk(f, a, b, j, k - 1, method)
    ) / (4**k - 1)


def __romberg_jk_print(
    f: Callable[[float], float],
    a: float,
    b: float,
    j: int,
    k: int,
    method: IntMethod,
    digits: int = 4,
    print_int_results=False,
) -> float:
    method_letter = "I"
    match method:
        case IntMethod.trapez:
            method_letter = "T"
        case IntMethod.rect:
            method_letter = "R"
        case IntMethod.simpson:
            method_letter = "S"
    if k == 0:
        h = (b - a) / 2**j
        result = 0
        match method:
            case IntMethod.rect:
                if print_int_results:
                    result = R_print(f, a, b, h, digits)
                else:
                    result = R(f, a, b, h)
            case IntMethod.trapez:
                if print_int_results:
                    result = T_print(f, a, b, h, digits)
                else:
                    result = T(f, a, b, h)
            case IntMethod.simpson:
                if print_int_results:
                    result = S_print(f, a, b, h, digits)
                else:
                    result = S(f, a, b, h)
        print(f"{method_letter}{j},{k}({h}) = {round(result, digits)}")
        return result

    result = (
        4**k * __romberg_jk_print(f, a, b, j + 1, k - 1, method, digits)
        - __romberg_jk_print(f, a, b, j, k - 1, method, digits)
    ) / (4**k - 1)

    print(
        f"{method_letter}{j},{k} = (4^{k}·{method_letter}{j + 1},{k - 1} - {method_letter}{j},{k - 1})/4^{k}-1 = {round(result, digits)}"
    )

    return result


# %%%%%%%%%%%%%%%%%%%%%%% ODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def euler(
    f: Callable[[float, float], float], a: float, b: float, h: float, y0: float
) -> Tuple[Vector, Vector]:
    """
    Solves the provided ODE `f` using Euler's method.\n
    p = 1

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    for i in range(n):
        x.append(x[i] + h)
        y.append(y[i] + h * f(x[i], y[i]))
    return x, y


def midpoint(
    f: Callable[[float, float], float], a: float, b: float, h: float, y0: float
) -> Tuple[Vector, Vector]:
    """
    Solves the provided ODE `f` using the Midpoint method.
    This method is more precise than Euler's and modified Euler's method.

    p = 2

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    for i in range(n):
        x.append(x[i] + h)
        xh2 = x[i] + (h / 2)
        yh2 = y[i] + ((h / 2) * f(x[i], y[i]))
        y.append(y[i] + h * f(xh2, yh2))
    return x, y


def modified_euler(
    f: Callable[[float, float], float], a: float, b: float, h: float, y0: float
) -> Tuple[Vector, Vector]:
    """
    Solves the provided ODE `f` using the modified Euler's method.

    p = 2

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    for i in range(n):
        x.append(x[i] + h)
        k1 = f(x[i], y[i])
        k2 = f(x[i + 1], y[i] + h * k1)
        y.append(y[i] + h * ((k1 + k2) / 2))
    return x, y


def runge_kutta_4(
    f: Callable[[float, float], float], a: float, b: float, h: float, y0: float
) -> Tuple[Vector, Vector]:
    """
    Solves the provided ODE `f` using the the 4-point Runge-Kutta method.

    p = 4

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    for i in range(n):
        x.append(x[i] + h)
        k1 = f(x[i], y[i])
        k2 = f(x[i] + (h / 2), y[i] + (h / 2) * k1)
        k3 = f(x[i] + (h / 2), y[i] + (h / 2) * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y.append(y[i] + h * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
    return x, y


def RK4(
    f: Callable[[float, float], float], a: float, b: float, h: float, y0: float
) -> Tuple[Vector, Vector]:
    """
    Solves the provided ODE `f` using the the 4-point Runge-Kutta method.

    p = 4

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value
    """
    return runge_kutta_4(f, a, b, h, y0)


def euler_print(
    f: Callable[[float, float], float],
    a: float,
    b: float,
    h: float,
    y0: float,
    digits: int = 4,
) -> Tuple[Vector, Vector]:
    """
    Solves the provided ODE `f` using Euler's method.\n
    p = 1

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value\n
    `digits` ⇒ Number of significant digits to round to when printing.
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    print("\n%%%%%%%%%%%%%%%%%%%% EULER %%%%%%%%%%%%%%%%%%%%\n")
    for i in range(n):
        print(f"========== Iteration {i} ==========")
        x_new = x[i] + h
        y_new = y[i] + h * f(x[i], y[i])
        x.append(x_new)
        y.append(y_new)
        print(f"x{i + 1} = x{i} + h")
        print(f"  = {round(x_new, digits)}")
        print(f"y{i + 1} = y{i} + h·f(x{i}, y{i})")
        print(f"  = y{i} + h·{round(f(x[i], y[i]), digits)}")
        print(f"  = {round(y_new, digits)}")
        print("========== End ==========", end="\n\n")
    print("%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%\n")
    return x, y


def midpoint_print(
    f: Callable[[float, float], float],
    a: float,
    b: float,
    h: float,
    y0: float,
    digits: int = 4,
) -> Tuple[Vector, Vector]:
    """
    Solves the provided ODE `f` using the Midpoint method.
    This method is more precise than Euler's and modified Euler's method.

    p = 2

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value\n
    `digits` ⇒ Number of significant digits to round to when printing.
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    print("\n%%%%%%%%%%%%%%%%%%%% MITTELPUNKT %%%%%%%%%%%%%%%%%%%%\n")
    for i in range(n):
        print(f"========== Iteration {i} ==========")
        x_new = x[i] + h
        xh2 = x[i] + (h / 2)
        yh2 = y[i] + ((h / 2) * f(x[i], y[i]))
        y_new = y[i] + h * f(xh2, yh2)
        x.append(x_new)
        y.append(y_new)
        print(f"x{i + 1} = x{i} + h")
        print(f"  = {round(x_new, digits)}")
        print(f"xh2 = x{i} + ½h")
        print(f"  = {round(xh2, digits)}")
        print(f"yh2 = y{i} + ½h·f(x{i}, y{i})")
        print(f"  = y{i} + ½h·{round(f(x[i], y[i]), digits)}")
        print(f"  = {round(yh2, digits)}")
        print(f"y{i + 1} = y{i} + h·f(xh2, yh2)")
        print(f"  = y{i} + h·{round(f(xh2, yh2), digits)}")
        print(f"  = {round(y_new, digits)}")
        print("========== End ==========", end="\n\n")
    print("%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%\n")
    return x, y


def modified_euler_print(
    f: Callable[[float, float], float],
    a: float,
    b: float,
    h: float,
    y0: float,
    digits: int = 4,
) -> Tuple[Vector, Vector]:
    """
    Solves the provided ODE `f` using the modified Euler's method.

    p = 2

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value\n
    `digits` ⇒ Number of significant digits to round to when printing.
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    print("\n%%%%%%%%%%%%%%%%%%%% MODIFIZIERTER EULER %%%%%%%%%%%%%%%%%%%%\n")
    for i in range(n):
        print(f"========== Iteration {i} ==========")
        x_new = x[i] + h
        x.append(x_new)
        k1 = f(x[i], y[i])
        k2 = f(x[i + 1], y[i] + h * k1)
        y_new = y[i] + h * ((k1 + k2) / 2)
        y.append(y_new)
        print(f"x{i + 1} = x{i} + h")
        print(f"  = {round(x_new, digits)}")
        print(f"k1 = f(x{i}, y{i})")
        print(f"  = {round(k1, digits)}")
        print(f"k2 = f(x{i + 1}, y{i} + h·k1)")
        print(f"  = {round(k2, digits)}")
        print(f"y{i + 1} = y{i} + h·½(k1+k2)")
        print(f"  = {round(y_new, digits)}")
        print("========== End ==========", end="\n\n")
    print("%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%\n")
    return x, y


def runge_kutta_4_print(
    f: Callable[[float, float], float],
    a: float,
    b: float,
    h: float,
    y0: float,
    digits: int = 4,
) -> Tuple[Vector, Vector]:
    """
    Solves the provided ODE `f` using the the 4-point Runge-Kutta method.

    p = 4

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value\n
    `digits` ⇒ Number of significant digits to round to when printing.
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    print("\n%%%%%%%%%%%%%%%%%%%% RUNGE-KUTTA 4 %%%%%%%%%%%%%%%%%%%%\n")
    for i in range(n):
        print(f"========== Iteration {i} ==========")
        x_new = x[i] + h
        x.append(x_new)
        k1 = f(x[i], y[i])
        k2 = f(x[i] + (h / 2), y[i] + (h / 2) * k1)
        k3 = f(x[i] + (h / 2), y[i] + (h / 2) * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y_new = y[i] + h * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y.append(y_new)
        print(f"x{i + 1} = x{i} + h")
        print(f"  = {round(x_new, digits)}")
        print(f"k1 = f(x{i}, y{i})")
        print(f"  = {round(k1, digits)}")
        print(f"k2 = f(x{i} + ½h, y{i} + ½h·k1)")
        print(
            f"  = f({round(x[i] + 0.5 * h, digits)}, {round(y[i] + 0.5 * h * k1, digits)})"
        )
        print(f"  = {round(k2, digits)}")
        print(f"k3 = f(x{i} + ½h, y{i} + ½h·k2)")
        print(
            f"  = f({round(x[i] + 0.5 * h, digits)}, {round(y[i] + 0.5 * h * k2, digits)})"
        )
        print(f"  = {round(k3, digits)}")
        print(f"k4 = f(x{i} + h, y{i} + h·k3)")
        print(f"  = f({round(x[i] + h, digits)}, {round(y[i] + h * k3)})")
        print(f"  = {round(k4, digits)}")
        print(f"y{i + 1} = y{i} + ⅙h(k1 + 2·k2 + 2·k3 + k4)")
        print(f"  = {round(y_new, digits)}")
        print("========== End ==========", end="\n\n")
    print("%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%\n")
    return x, y


def RK4_print(
    f: Callable[[float, float], float],
    a: float,
    b: float,
    h: float,
    y0: float,
    digits: int = 4,
) -> Tuple[Vector, Vector]:
    """
    Solves the provided ODE `f` using the the 4-point Runge-Kutta method.

    p = 4

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value\n
    `digits` ⇒ Number of significant digits to round to when printing.
    """
    return runge_kutta_4_print(f, a, b, h, y0)


def runge_kutta_8(
    f: Callable[[float, float], float], a: float, b: float, h: float, y0: float
) -> Tuple[Vector, Vector]:
    """
    Solves the provided ODE `f` using the the 8-point Runge-Kutta method.

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    for i in range(n):
        x.append(x[i] + h)
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h * (4 / 27), y[i] + h * (4 / 27) * k1)
        k3 = f(x[i] + h * (2 / 9), y[i] + (h / 18) * (k1 + 3 * k2))
        k4 = f(x[i] + h * (1 / 3), y[i] + (h / 12) * (k1 + 3 * k3))
        k5 = f(x[i] + h * (1 / 2), y[i] + (h / 8) * (k1 + 3 * k4))
        k6 = f(
            x[i] + h * (2 / 3), y[i] + (h / 54) * (13 * k1 - 27 * k3 + 42 * k4 + 8 * k5)
        )
        k7 = f(
            x[i] + h * (1 / 6),
            y[i] + (h / 4320) * (389 * k1 - 54 * k3 + 966 * k4 - 824 * k5 + 243 * k6),
        )
        k8 = f(
            x[i] + h,
            y[i]
            + (h / 20)
            * (-234 * k1 + 81 * k3 - 1164 * k4 + 656 * k5 - 122 * k6 + 800 * k7),
        )
        k9 = f(
            x[i] + h * (5 / 6),
            y[i]
            + (h / 288)
            * (-127 * k1 + 18 * k3 - 678 * k4 + 456 * k5 - 9 * k6 + 576 * k7 + 4 * k8),
        )
        k10 = f(
            x[i] + h,
            y[i]
            + (h / 820)
            * (
                1481 * k1
                - 81 * k3
                + 7104 * k4
                - 3376 * k5
                + 72 * k6
                - 5040 * k7
                - 60 * k8
                + 720 * k9
            ),
        )
        y.append(
            y[i]
            + (h / 840)
            * (41 * k1 + 27 * k4 + 272 * k5 + 27 * k6 + 216 * k7 + 216 * k9 + 41 * k10)
        )
    return x, y


def RK8(
    f: Callable[[float, float], float], a: float, b: float, h: float, y0: float
) -> Tuple[Vector, Vector]:
    """
    Solves the provided ODE `f` using the the 8-point Runge-Kutta method.

    p = ?

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value
    """
    return runge_kutta_8(f, a, b, h, y0)


# %%%%%%%%%%%%%%%%%%%%%%% SYSTEMS OF ODES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def euler_vec(
    f: Callable[[float, Vector], Vector], a: float, b: float, h: float, y0: Vector
) -> Tuple[Vector, Matrix]:
    """
    Solves the provided system of ODEs `f` using Euler's method.\n
    p = 1

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    for i in range(n):
        x.append(x[i] + h)
        # y⁽ⁱ⁺¹⁾ = y⁽ⁱ⁾ + h·f(xᵢ, y⁽ⁱ⁾)
        y_new = [y + h * n for y, n in zip(y[i], f(x[i], y[i]))]
        y.append(y_new)
    return x, y


def midpoint_vec(
    f: Callable[[float, Vector], Vector], a: float, b: float, h: float, y0: Vector
) -> Tuple[Vector, Matrix]:
    """
    Solves the provided system of ODEs `f` using the Midpoint method.
    This method is more precise than Euler's and modified Euler's method.

    p = 2

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    for i in range(n):
        x.append(x[i] + h)
        xh2 = x[i] + (h / 2)
        yh2 = [y + (h / 2) * n for n, y in zip(f(x[i], y[i]), y[i])]
        y.append([y + h * n for y, n in zip(y[i], f(xh2, yh2))])
    return x, y


def modified_euler_vec(
    f: Callable[[float, Vector], Vector], a: float, b: float, h: float, y0: Vector
) -> Tuple[Vector, Matrix]:
    """
    Solves the provided system of ODEs `f` using the modified Euler's method.

    p = 2

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    for i in range(n):
        x.append(x[i] + h)
        k1 = f(x[i], y[i])
        k2 = f(x[i + 1], [y + h * n for y, n in zip(y[i], f(x[i], y[i]))])
        y.append([y + h * (k1 + k2) / 2 for y, k1, k2 in zip(y[i], k1, k2)])
    return x, y


def runge_kutta_4_vec(
    f: Callable[[float, Vector], Vector], a: float, b: float, h: float, y0: Vector
) -> Tuple[Vector, Matrix]:
    """
    Solves the provided system of ODEs `f` using the the 4-point Runge-Kutta method.

    p = 4

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value
    """
    n = int(math.ceil((b - a) / h))
    x = [a]
    y = [y0]
    for i in range(n):
        x.append(x[i] + h)
        k1 = f(x[i], y[i])
        k2 = f(x[i] + (h / 2), [y + (h / 2) * k1 for y, k1 in zip(y[i], k1)])
        k3 = f(x[i] + (h / 2), [y + (h / 2) * k2 for y, k2 in zip(y[i], k2)])
        k4 = f(x[i] + h, [y + h * k3 for y, k3 in zip(y[i], k3)])
        y.append(
            [
                y + h * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                for y, k1, k2, k3, k4 in zip(y[i], k1, k2, k3, k4)
            ]
        )
    return x, y


def RK4_vec(
    f: Callable[[float, Vector], Vector], a: float, b: float, h: float, y0: Vector
) -> Tuple[Vector, Matrix]:
    """
    Solves the provided system of ODEs `f` using the the 4-point Runge-Kutta method.

    p = 4

    ---
    `f` ⇒ 1st order derivative (ODE)\n
    `a` ⇒ Lower x bound\n
    `b` ⇒ Upper x bound\n
    `h` ⇒ Precision\n
    `y0` ⇒ Starting value
    """
    return runge_kutta_4_vec(f, a, b, h, y0)
