import num
import math
from scipy import integrate


def f(t: float) -> float:
    return 2 * math.exp(-(((t / 10) - 2) ** 4))


a = 0
b = 40

h0 = 40 / 2**0
h1 = 40 / 2**1
h2 = 40 / 2**2
h3 = 40 / 2**3

print("T0,0 =")
num.T_print(f, a, b, h0)
print("T1,0 =")
num.T_print(f, a, b, h1)
print("T2,0 =")
num.T_print(f, a, b, h2)
print("T3,0 =")
num.T_print(f, a, b, h3)

integrate.romberg(f, a, b, show=True, divmax=3)
