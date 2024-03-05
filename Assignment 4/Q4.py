from scipy.optimize import least_squares
import numpy as np

import matplotlib.pyplot as plt

x, y = np.loadtxt("periodic.dat").T

# n = 3, last element is the constant
c = np.array([1, 1, 1, 1])

def b(a, x):
    return np.cos(2*np.pi*a*x)

def f(x, c):
    *c, const = c
    b_a = np.array([b(alpha, x) for alpha in range(len(c))])
    return np.exp(np.sum(np.diag(c)@b_a, axis=0)) + const

def chi(c):
    return y-f(x, c)

r = least_squares(chi, c, method="lm")

plt.plot(x, y)
plt.plot(x, f(x, r.x))
plt.show()