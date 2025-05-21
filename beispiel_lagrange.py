from typing import List
import num
from num import Vector
import matplotlib.pyplot as plt
import numpy as np

x_vals: Vector = [1, 2, 3, 4]
y_vals: Vector = [3, 4, 1, 2]

p = np.polyfit(x_vals, y_vals, len(x_vals) - 1)

x = np.linspace(0, 5, 100)
y = [np.polyval(p, x) for x in x]
y_own = num.lagrange_vec(list(x), x_vals, y_vals)

x_5 = num.lagrange_point(5, x_vals, y_vals, printing=True)
print(f"P(5) = {x_5}")

plt.scatter(x_vals, y_vals, label="data", color="lime", marker=".", zorder=2)
plt.plot(x, y, linestyle="dotted", label="np.polyfit", zorder=1)
plt.plot(x, y_own, label="own lagrange", zorder=0)
# Empty plot for additional legend entry
# plt.plot([], [], " ", label="$y = ax^3+bx^2+cx+d$")
plt.legend()
plt.show()
