import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1.0/(1.0+10.0*x*x)

def cheb_coeff(points, f):
    A = np.polynomial.chebyshev.chebvander(points, len(points)-1)
    c = f(points)
    b = np.linalg.solve(A, c)
    return b

def cheb_eval(x, coefficients):
    return np.polynomial.chebyshev.chebval(x, coefficients)

def print_coefficients(coefficients):
    for i, coefficient in enumerate(coefficients):
        print(f"\t{coefficient:+.2f}\tT_{i}")

if __name__ == "__main__":
    for N in [10, 100]:
        start = -1
        stop = 1

        points = np.linspace(start, stop, N)
        coefficients = cheb_coeff(points, f)

        x = np.linspace(start, stop, 100)
        y = cheb_eval(x, coefficients)

        print(f"Coefficients for {N} sample points")
        print_coefficients(coefficients)       

        plt.plot(x, y, label="Approximation")
        plt.plot(x, f(x), label="Exact")
        plt.title(f"Approximation using {N} points.")
        plt.legend()
        plt.show()
    
