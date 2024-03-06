from scipy.optimize import least_squares
import numpy as np

def optimize(x, y, c):
    """
    x: np.array (l)
    y: np.array (l)
    c: np.array (n)
    basis: function (n),(l)->(n, l)
    """
    @np.vectorize(signature='(n),(m)->(n, m)')
    def basis(a, x):
        return np.cos(2*np.pi*np.outer(a, x))

    def f(x, c):
        *c, const = c
        return (
            np.exp(
                np.sum(
                    np.diag(c)@basis(np.arange(len(c)), x),
                    axis=0
                )
            ) + const
        )

    def residuals(c):
        return y-f(x, c)

    r = least_squares(residuals, c, method="lm")

    return x, f(x, r.x)
