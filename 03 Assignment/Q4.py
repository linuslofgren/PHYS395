import numpy as np
from numpy.random import normal, standard_cauchy
from numpy.polynomial.legendre import legvander
import matplotlib.pyplot as plt

x1, x2 = np.loadtxt('samples.dat').T

n = len(x1)

cdf1 = np.sort(x1)
cdf2 = np.sort(x2)

KSdistance = np.max(np.abs(cdf1 - cdf2))
print(f"Kolmogorov-Smirnov distance is {KSdistance:.2f}")

# Constant for 95% confidence
a = 1.358

rejection_level =  a * np.sqrt(2*n/(n*n))

print("Not the same distribution at 95% confidence level" if KSdistance > rejection_level else "Same distribution at 95% confidence level")