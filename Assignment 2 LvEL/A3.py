import numpy as np
import matplotlib.pyplot as plt
from A1 import f, cheb_coeff, cheb_eval, print_coefficients
from A2 import cheb_deriv
# def f(x):
#     return 1.0/(1.0+10.0*x*x)

def fprime(x):
    return -(20.0*x)/ (1.0 + 10.0 * x*x) **2

def compare_derivative_approx(sample_points, evaluation_points):
    f_prime_cheb_coeff = cheb_deriv(sample_points, f)

    print("="*10+"Coefficients for f' "+"="*10)
    print_coefficients(f_prime_cheb_coeff)

    f_prime_cheb = cheb_eval(evaluation_points, f_prime_cheb_coeff)

    f_prime_exact = fprime(evaluation_points)

    return f_prime_cheb, f_prime_exact

def compare_approx(sample_points, evaluation_points):
    f_cheb_coeff = cheb_coeff(sample_points, f)

    print("="*10+"Coefficients for f "+"="*10)
    print_coefficients(f_cheb_coeff)

    f_cheb = cheb_eval(evaluation_points, f_cheb_coeff)

    f_exact = f(evaluation_points)

    return f_cheb, f_exact

if __name__ == "__main__":
    for N in [10, 100]:
        x = np.linspace(-1, 1, 100)
        approx, exact = compare_derivative_approx(np.linspace(-1, 1, N), x)
        plt.plot(x, np.abs(approx-exact))
        plt.title(f"Error for derivative with {N} points")
        plt.show()
