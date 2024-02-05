import numpy as np
import matplotlib.pyplot as plt
from problem_4 import spline

def error_for(N):
    x = np.linspace(0, 2*np.pi, 100, endpoint=False)[1:]
    c = spline(N=N, start=0, end=2*np.pi)

    error = np.abs(c(x) - np.sin(x))
    return error, x

def plot_error(linspace, error, output):
    max_error_index = np.argmax(error)
    max_error = error[max_error_index]

    _, ax = plt.subplots()
    ax.set_title(f"Maximum error was {max_error:.4f} at x={linspace[max_error_index]:.4f}")
    ax.plot(linspace, error, "r-", alpha=0.8, label="Error")
    ax.plot(linspace[max_error_index], max_error, "ro", alpha=0.4)
    ax.legend(loc='lower left')
    plt.savefig(output)

def run():
    error, linspace = error_for(10)
    plot_error(linspace, error, "output/problem_5_10pts.pdf")

    error, linspace = error_for(20)
    plot_error(linspace, error, "output/problem_5_20pts.pdf")
    

if __name__ == "__main__":
    run()
