import numpy as np
import matplotlib.pyplot as plt
from A1 import f, cheb_coeff, cheb_eval
# def f(x):
#     return 1.0/(1.0+10.0*x*x)

def fprime(x):
    return -(20.0*x)/ (1.0 + 10.0 * x*x) **2

def cheb_deriv(sample_points, f, order=1):
    coefficients = cheb_coeff(sample_points, f)
    derivative_coefficients = np.polynomial.chebyshev.chebder(coefficients, m=order)
    return derivative_coefficients


if __name__ == "__main__":
    for N in [10, 100]:
        sample_points = np.linspace(-1, 1, N)
        coeff = cheb_deriv(sample_points, f)

        x = np.linspace(-1, 1, 100)
        y = cheb_eval(x, coeff)

        plt.plot(x, y)
        plt.plot(x, fprime(x))
        plt.legend()
        plt.title(f"Derivative with {N} points")
        plt.show()
