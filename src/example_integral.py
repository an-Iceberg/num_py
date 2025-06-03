from math import cos, sin
from integral import int_2, int_3, int_4


def f(x: float) -> float:
    return cos(x)


def F(x: float) -> float:
    return sin(x)


a = 0.0
b = 5.0

print(f"Simpson's ⅓: ε = {abs(int_2(a, b, f) - (F(b) - F(a))):.2e}")
print(f"Simpson's ⅜: ε = {abs(int_3(a, b, f, 0.00001) - (F(b) - F(a))):.2e}")
print(f"Boole's :    ε = {abs(int_4(a, b, f) - (F(b) - F(a))):.2e}")
