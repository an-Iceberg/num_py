from typing import List
import num

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# plt.style.use("dracula")

x_vals: List[float] = [1, 2, 3, 4, 5, 6, 7]
y_vals: List[float] = [1, 2, 2, 1, 2, 2, 1]
spline = interpolate.CubicSpline(x_vals, y_vals, bc_type="natural")

x = np.linspace(-1.5, 9.5, 150)
y_own = num.spline_print(list(x), x_vals, y_vals, digits=5)

# plt.scatter(x_vals, y_vals, label="data", color="#ffb86c", marker=".", zorder=1)
plt.scatter(x_vals, y_vals, label="data", color="magenta", marker=".", zorder=2)
plt.plot(x, spline(x), linestyle="dotted", label="scipy spline", zorder=1)
plt.plot(x, y_own, label="own spline", zorder=0)
plt.legend()
plt.show()
