import matplotlib.pyplot as plt
import numpy as np
from Q4 import optimize

x, y = np.loadtxt("periodic.dat").T

# n = 3, last element is the constant
c = np.array([1, 1, 1, 1])
x_fit, y_fit = optimize(x, y, c)

fig, ax = plt.subplots()
ax.plot(x, y, label="Data from Assignment #2")
ax.plot(x_fit, y_fit, label="Fitted model")
ax.legend()
ax.set_title("Fitted $f(x)=exp[\\sum_{\\alpha}c_{\\alpha}b_{\\alpha}(x)]+const$ with $b_{\\alpha}(x)=cos(2\\pi x)$, $\\alpha=0...3$.")
fig.savefig("Q5.pdf")
plt.show()