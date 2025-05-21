import num
import math
import matplotlib.pyplot as plt

# Beispiel 8.9 & 8.10 aus dem Skript


def f(x: float, z: num.Vector):
    return [
        z[1],
        z[2],
        10 * math.exp(-x) - 5 * z[2] - 8 * z[1] - 6 * z[0],
    ]


z0: num.Vector = [2, 0, 0]

a = 0
b = 1
h = 0.5

x, y = num.euler_vec(f, a, b, h, z0)
# x, y = num.midpoint_vec(f, a, b, h, z0)

# y0 = [yi[0] for yi in y]
# y1 = [yi[1] for yi in y]
# y2 = [yi[2] for yi in y]

print(f"x = {[round(n, 2) for n in x]}")
print("y =")
[print(f"  {[round(n, 2) for n in row]}") for row in y]

# plt.plot(x, y0)
# plt.plot(x, y1)
# plt.plot(x, y2)
plt.plot(x, y)
plt.legend(["${z_1}'$", "${z_2}'$", "${z_3}'$"])
plt.show()
