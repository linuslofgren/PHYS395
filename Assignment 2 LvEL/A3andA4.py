import numpy as np
import matplotlib.pyplot as plt
from A3 import compare_approx, compare_derivative_approx
from A1 import print_coefficients

def max_w_index(l):
    i = np.argmax(l)
    return i, l[i]

if __name__ == "__main__":
    for N in [10, 100]:
        print("#"*20+f"\tEvaluating N={N}\t"+"#"*20)
        x = np.linspace(-1, 1, 100)

        print("Linear approximation of f")
        approx_linear, exact = compare_approx(np.linspace(-1, 1, N), x)
        print("Root approximation of f")
        approx_roots, _ = compare_approx(np.cos(np.pi/N*(np.arange(N)+0.5)), x)

        print("Linear approximation of f")
        approx_derivative_linear, exact_derivative = compare_derivative_approx(np.linspace(-1, 1, N), x)
        print("Root approximation of f'")
        approx_derivative_roots, _ = compare_derivative_approx(np.cos(np.pi/N*(np.arange(N)+0.5)), x)

        fig, ((f_ax, f_error_ax), (f_prime_ax, f_prime_error_ax)) = plt.subplots(2, 2, figsize=(14, 8))

        f_ax.plot(x, approx_linear, label="$f(x)_{l}$ Linear sampling")
        f_ax.plot(x, approx_roots, label="$f(x)_{r}$ Root sampling")
        f_ax.plot(x, exact, label="$f(x)$ Exact")
        f_ax.set_title(r"$f(x)=\frac{1}{1+10x^2}$")

        f_error_ax.plot(x, np.abs(approx_linear-exact), label="Linear sampling")
        f_error_ax.plot(x, np.abs(approx_roots-exact), label="Root sampling")

        max_lin_error_index, max_lin_error = max_w_index(np.abs(approx_linear-exact))
        max_root_error_index, max_root_error = max_w_index(np.abs(approx_roots-exact))
        f_error_ax.set_title(
            r"$\mathrm{sup}|f(x)-f(x)_l|=$"
            +f"{max_lin_error:.2f} at x={x[max_lin_error_index]:.2f}, "
            + r"$\mathrm{sup}|f(x)-f(x)_r|=$"
            +f"{max_root_error:.2f} at x={x[max_root_error_index]:.2f}"
        )

        f_prime_ax.plot(x, approx_derivative_linear, label="Linear sampling")
        f_prime_ax.plot(x, approx_derivative_roots, label="Root sampling")
        f_prime_ax.plot(x, exact_derivative, label="Exact")
        f_prime_ax.set_title(r"$\frac{\mathrm{d} f(x)}{\mathrm{d}x}=\frac{-20x}{(1+10x^2)^2}$")

        f_prime_error_ax.plot(x, np.abs(approx_derivative_linear-exact_derivative), label="Linear sampling")
        f_prime_error_ax.plot(x, np.abs(approx_derivative_roots-exact_derivative), label="Root sampling")

        max_lin_der_error_index, max_lin_der_error = max_w_index(np.abs(approx_derivative_linear-exact_derivative))
        max_root_der_error_index, max_root_der_error = max_w_index(np.abs(approx_derivative_roots-exact_derivative))
        f_prime_error_ax.set_title(
            r"$\mathrm{sup}|\frac{\mathrm{d} f(x)}{\mathrm{d}x}-\frac{\mathrm{d} f(x)_l}{\mathrm{d}x}|=$"
            +f"{max_lin_der_error:.2f} at x={x[max_lin_der_error_index]:.2f}, "
            + r"$\mathrm{sup}|\frac{\mathrm{d} f(x)}{\mathrm{d}x}-\frac{\mathrm{d} f(x)_r}{\mathrm{d}x}|=$"
            +f"{max_root_der_error:.2f} at x={x[max_root_der_error_index]:.2f}"
        )

        handles, labels = f_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')
        fig.suptitle(f"Function approximation using  N={N} linear and root sampling")
        plt.show()
        fig.savefig(f"A3 and A4 N={N}.pdf")