import numpy as np
import matplotlib.pyplot as plt
from problem_4 import spline

def run():
    N = 20
    x = np.linspace(0, 2*np.pi, 100, endpoint=False)[1:]
    c = spline(N=N, start=0, end=2*np.pi)

    diff = np.abs(c(x) - np.sin(x))
    max_error_index = np.argmax(diff)
    max_error = diff[max_error_index]

    _, ax = plt.subplots()
    ax.plot(x, np.sin(x), label="sin")
    ax.plot(x, c(x), label="Cubic Spline")
    ax.set_title(f"N={N}, Max Error={max_error:.3f}")
    ax.legend(loc='lower left')

    ax2 = ax.twinx()
    ax2.plot(x, diff, "r-", alpha=0.2)    
    ax2.plot(x[max_error_index], diff[max_error_index], "ro", alpha=0.4)

    plt.savefig("output/problem_5.pdf")

if __name__ == "__main__":
    run()
