import num
import math

import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return np.cos(x**2) * np.sin(y)


xmin = 0
xmax = 20
ymin = 0
ymax = 3
hx = 0.2
hy = 0.25

nx: int = int(math.ceil((xmax - xmin) / hx))
x = np.linspace(xmin, xmax, nx)
ny: int = int(math.ceil((ymax - ymin) / hy))
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)
Ydiff = f(X, Y)


plt.style.use("dracula")

width = 0.25

# plt.quiver(X, Y, np.ones(np.shape(Ydiff)), Ydiff, color="#a4a4a4", width=0.001)
x, y = num.runge_kutta_8(f, xmin, xmax, 0.01, 1)
plt.plot(x, y, label="«exact» RK8 h=0.01")
x, y = num.runge_kutta_8(f, xmin, xmax, width, 1)
plt.plot(x, y, label="RK8")
x, y = num.runge_kutta_4(f, xmin, xmax, width, 1)
plt.plot(x, y, label="RK4")
x, y = num.modified_euler(f, xmin, xmax, width, 1)
plt.plot(x, y, label="modified euler")
x, y = num.midpoint(f, xmin, xmax, width, 1)
plt.plot(x, y, label="midpoint")
x, y = num.euler(f, xmin, xmax, width, 1)
plt.plot(x, y, label="euler")
plt.legend()
plt.show()

# _, _ = num.euler_print(f, xmin, xmax, 1, 1)
# _, _ = num.midpoint_print(f, xmin, xmax, 1, 1)
# _, _ = num.modified_euler_print(f, xmin, xmax, 1, 1)
# _, _ = num.RK4_print(f, xmin, xmax, 1, 1)
