import numpy as np
from numpy.random import normal, standard_cauchy
from numpy.polynomial.legendre import legvander
import matplotlib.pyplot as plt
from Q2 import get_pdf, generate

N = 1_024
runs = 10_000

samples = generate((N, runs))
y = np.sort(samples, axis=0)
n = len(y)

P = legvander(np.linspace(-1.0, 1.0, n), 3)

L1, L2, L3, L4 = [P[:,k-1]@y/n for k in (1, 2, 3, 4)]
M2, M3, M4 = [np.sum(np.power(y, k), axis=0)/n for k in (2, 3, 4)]

skew_L = L3/L2
skew_M = M3/(np.power(M2, 3.0/2.0))

kurtosis_L = L4/L2
kurtosis_m = M4/np.power(M2, 2)

names = ("Skew L", "Skew M", "Kurtosis L", "Kurtosis M")
data = (skew_L, skew_M, kurtosis_L, kurtosis_m)

fig, (axis) = plt.subplots(1, 4, figsize=(14, 8))

for name, ax, data in zip(names, axis, data):
    x, pdf = get_pdf(data, n=40)
    ax.plot(x, pdf)
    ax.set_title(name)

print("None of the skewness estimators have a pdf which goes below 0.")
print("None of the kurtosis estimators have the wrong sign of excess.")

plt.show()
fig.savefig("Q3.pdf")