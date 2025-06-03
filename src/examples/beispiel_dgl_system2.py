import math
import matplotlib.pyplot as plt

from num import Vector
from num import euler_vec
from num import midpoint_vec
from num import modified_euler_vec
from num import RK4_vec

# Beispiel 8.9 & 8.10 aus dem Skript


def f(x: float, z: Vector):
    z1, z2, z3 = z
    return [
        z2,
        z3,
        10 * math.exp(-x) - 5 * z3 - 8 * z2 - 6 * z1,
    ]


z_start: Vector = [2, 0, 0]

a = 0
b = 7
h = 0.01

plt.style.use("dracula")

########## Euler #########

x, y = euler_vec(f, a, b, h, z_start)

plt.subplot(2, 2, 1)
plt.title("Verfahren: Euler")
plt.plot(x, y)
plt.legend(["$y$", "$y'$", "$y''$"])

########## Mittelpunkt #########

x, y = midpoint_vec(f, a, b, h, z_start)

plt.subplot(2, 2, 2)
plt.title("Verfahren: Mittelpunkt")
plt.plot(x, y)
plt.legend(["$y$", "$y'$", "$y''$"])

########## Modifizerter Euler #########

x, y = modified_euler_vec(f, a, b, h, z_start)

plt.subplot(2, 2, 3)
plt.title("Verfahren: Modifizerter Euler")
plt.plot(x, y)
plt.legend(["$y$", "$y'$", "$y''$"])

########## Runge Kutta 4 #########

x, y = RK4_vec(f, a, b, h, z_start)

# y0 = [yi[0] for yi in y]
# y1 = [yi[1] for yi in y]
# y2 = [yi[2] for yi in y]

plt.subplot(2, 2, 4)
plt.title("Verfahren: Runge Kutta 4")
plt.plot(x, y)
plt.legend(["$y$", "$y'$", "$y''$"])

plt.tight_layout()
plt.show()
