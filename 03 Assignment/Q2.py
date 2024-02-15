import numpy as np
from numpy.random import normal, standard_cauchy
from numpy.polynomial.legendre import legvander
import matplotlib.pyplot as plt

def generate(N):
    x = normal(size=N)
    return x + np.power(x, 2)/6

N = 10000
samples = np.sort(generate(N))

# Generate the target evaluation points
rank = np.linspace(0.0, 1.0, len(samples))
# Number of interpolation points
n = 24
# Number of interpolation points
theta = np.linspace(np.pi, 0.0, n)
cdf = (np.cos(theta) + 1.0)/2.0
x = np.interp(cdf, rank, samples)

pdf = np.gradient(cdf, x, edge_order=2)

fig, ((f_1)) = plt.subplots(1, 1, figsize=(14, 8))
f_1.plot(x, pdf, 'r-')
f_1.set_title(r"Probability density function for $y=x+\frac{x^2}{6}$"+ f" using N={N} samples")
plt.show()
fig.savefig("Q2.pdf")
