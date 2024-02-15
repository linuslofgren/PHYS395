import numpy as np
from numpy.random import normal, standard_cauchy
from numpy.polynomial.legendre import legvander
import matplotlib.pyplot as plt

N = 8192

gaussian = normal(size=N)
cauchy = standard_cauchy(size=N)

def statistics(samples):
    n = len(samples)
    mean = np.mean(samples)
    median = np.median(samples)
    std = np.sqrt(np.mean(samples**2))
    k=2
    y = np.sort(samples); P = legvander(np.linspace(-1.0, 1.0, n), k)
    L2 = np.sum((y.T*P[:,k-1]).T)/n

    return mean, median, std, L2

gaussian_data = []
cauchy_data = []
for i in range(1, N):
    gaussian_data.append(statistics(gaussian[:i]))
    cauchy_data.append(statistics(cauchy[:i]))


fig, ((f_1, f_2)) = plt.subplots(1, 2, figsize=(14, 8))

f_1.plot(gaussian_data, label=["mean", "median", "std", "L2"])
f_1.set_title("Gauss")
f_1.set_xscale('log')
f_1.legend()
f_2.plot(cauchy_data, label=["mean", "median", "std", "L2"])
f_2.set_title("Cauchy")
f_2.set_xscale('log')
f_2.legend()
plt.savefig("Q1.pdf")