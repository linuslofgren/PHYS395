import numpy as np
from numpy.random import normal, standard_cauchy
from numpy.polynomial.legendre import legvander
import matplotlib.pyplot as plt


N = 1024
x = normal(size=(N, 10_000))
y = x + np.power(x, 2)/6


histogram = np.histogram(y, density=True)


def statistics(samples):
    n = len(samples)
    m2, m3, m4 = [np.sum(np.power(samples, k), axis=0)/n for k in (2, 3, 4)]
    k_max=4
    y = np.sort(samples);
    P = legvander(np.linspace(-1.0, 1.0, n), k_max)
    L2, L3, L4 = [np.sum((y.T*P[:,k-1]).T, axis=0)/n for k in (2, 3, 4)]


    skew_m = m3/(m2**(3.0/2.0))
    kurtosis_m = m4/(m2**2)

    skew_L = L3/L2
    kurtosis_L = L4/L2

    return skew_m, kurtosis_m, skew_L, kurtosis_L

fig, ((f_1, f_2, f_3, f_4)) = plt.subplots(1, 4, figsize=(14, 8))

skew_m, kurtosis_m, skew_L, kurtosis_L = statistics(y)

f_1.hist(skew_m, density=True, bins='auto', label="skew_m")
f_1.set_title("skew_m")
f_1.legend()

f_2.hist(kurtosis_m, density=True, bins='auto', label="kurtosis_m")
f_2.set_title("kurtosis_m")
f_2.legend()

print(np.min(skew_m), np.max(skew_m), np.mean(skew_m))
print(np.min(skew_L), np.max(skew_L), np.mean(skew_L))
f_3.hist(skew_L, density=True, bins='auto', label="skew_L")
f_3.set_title("skew_L")
f_3.legend()

f_4.hist(kurtosis_L, density=True, bins='auto', label="kurtosis_L")
f_4.set_title("kurtosis_L")
f_4.legend()

plt.show()
fig.savefig("Q3.pdf")