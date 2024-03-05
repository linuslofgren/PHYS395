import numpy as np
import matplotlib.pyplot as plt
from Q2 import get_cdf


def CalcKSDistance(cdf1, x1, cdf2, x2):
    overlap_from = np.max([np.min(x1), np.min(x2)])
    overlap_to = np.min([np.max(x1), np.max(x2)])
    grid = np.linspace(overlap_from, overlap_to, 1000)
    cdf1_normalized = np.interp(grid, x1, cdf1)
    cdf2_normalized = np.interp(grid, x2, cdf2)

    # plt.plot(grid, cdf1_normalized, label="normalized cdf 1")
    # plt.plot(grid, cdf2_normalized, label="normalized cdf 2")

    return np.max(np.abs(cdf1_normalized - cdf2_normalized))

if __name__ == "__main__":
    sample1, sample2 = np.loadtxt('samples.dat').T

    x1, cdf1 = get_cdf(sample1)
    x2, cdf2 = get_cdf(sample2)

    KSdistance = CalcKSDistance(cdf1, x1, cdf2, x2)
    # plt.plot(x1, cdf1, label="cdf 1")
    # plt.plot(x2, cdf2, label="cdf 2")
    # plt.legend()

    print(f"Kolmogorov-Smirnov distance is {KSdistance}")

    # Constant for 99% confidence
    a = 1.628
    n = len(sample1)
    rejection_level =  a* np.sqrt(2*n/(n*n))
    print("Null hypothesis rejected at 99% confidence level (can say that the distributions are different)" if KSdistance > rejection_level else "Null hypothesis not rejected at 99% confidence level (cannot say that the distributions are different)")

    # plt.show()