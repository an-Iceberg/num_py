from math import cos, sin
from integral import int_2, int_3, int_4, int_6


def f(x: float) -> float:
    return cos(x)


def F(x: float) -> float:
    return sin(x)


a = 0.0
b = 5.0

print(f"Simpson's ⅓: ε = {abs(int_2(a, b, f) - (F(b) - F(a))):.2e}")
print(f"Simpson's ⅜: ε = {abs(int_3(a, b, f) - (F(b) - F(a))):.2e}")
print(f"Boole's :    ε = {abs(int_4(a, b, f) - (F(b) - F(a))):.2e}")
print(f"Weddle's :   ε = {abs(int_6(a, b, f) - (F(b) - F(a))):.2e}")

# Why is Simpson's 1/3 & Weddle's so imprecise? :( What did i do wrong? :(
"""
Output:
Simpson's ⅓: ε = 5.33e-11
Simpson's ⅜: ε = 5.48e-03
Boole's :    ε = 2.11e-15
Weddle's :   ε = 5.48e-03
"""
