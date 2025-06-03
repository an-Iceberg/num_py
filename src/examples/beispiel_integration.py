import num
import math


def f(x: float) -> float:
    return 1 / x


ε = 1e-5
depth = 5

print("Trapez =   ", num.T(f, 1, 3, ε))
print("Rectangle =", num.R(f, 1, 3, ε))
print("Simpson =  ", num.S(f, 1, 3, ε))
print(
    "Romberg(T) =",
    num.romberg_print(f, 1, 2, depth, num.IntMethod.trapez, 4, False),
)
# If you want all intermediate results and steps, set last argument to «True»
# print(
#     "Romberg(T):",
#     num.romberg_print(f, 1, 2, depth, num.IntMethod.trapez, 4, True),
# )
print()
print("Romberg(R) =", num.romberg(f, 1, 3, depth, num.IntMethod.rect))
print("Romberg(S) =", num.romberg(f, 1, 3, depth, num.IntMethod.simpson))
print("Exact =     ", math.log(3))
print()


def g(x: float) -> float:
    return math.exp(x) - 0.8 * x**3 + 0.5 * math.sin(7 * x)


print(f"Tg({ε}) =", num.rect(g, 0, 4, ε))
print(f"Sg({ε}) =", num.simpson(g, 0, 4, ε))
print()
num.R_print(g, 0, 4, 0.5)
print()
num.T_print(g, 0, 4, 0.5)
print()
num.S_print(g, 0, 4, 0.5)
