import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def spline(N, start, end, function=np.sin):
    spline_x = np.linspace(start, end, N, endpoint=False)[1:]
    spline_y = function(spline_x)
    c = CubicSpline(spline_x, spline_y)
    return c

def run():
    x = np.linspace(0, 2*np.pi, 100, endpoint=False)[1:]
    y = np.sin(x)
    c = spline(N=10, start=0, end=2*np.pi)
    _, ax = plt.subplots()
    ax.plot(x, y, label="sin")
    ax.plot(x, c(x), label="cubic")
    ax.legend(loc='lower left', ncol=2)
    plt.savefig("output/problem_4.pdf")

    
if __name__ == "__main__":
    run()
