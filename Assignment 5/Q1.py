from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def evolution(l=1.0):
    def E(state):
        x, v = state

        return v**2/2.0 + l*x**4/4.0

    def f(t, state):
        x, v = state

        return np.array([v,-l*x**3])
    
    return E, f


def harmonic_series(t, n):
    gamma_forth = gamma(1/4)
    n = np.arange(1, n+1)
    T = np.pi**(-1/2)*gamma_forth**2
    return np.sum(2**(1/2)*4*np.pi/T*np.cos(np.outer((n-1/2)*4*np.pi/T, t))/np.cosh((n-1/2)*np.pi)[:,None], axis=0)

if __name__ == "__main__":

    for l, c in zip([1], ["-", "+", "."]):
        E, f = evolution(1.0)

        tmax = 10
        n = 1024
        t = np.linspace(0.0, tmax, n)

        state = np.array([l,0.0]); E0 = E(state)
        tol = 1e-12
        soln = solve_ivp(f, [t[0],t[-1]], state, method='Radau', t_eval=t, atol=1e-12)


        #plt.plot(soln.t, soln.y[0])
        x, v = soln.y
        # scale = l
        # n = 2
        # series = 2**(1/2)*4*np.pi*sum(map(lambda n: np.cos(n-1/2)*, range(0, n)))
        # for n in range(1, 100):
        ham = harmonic_series(soln.t, 100)
        plt.plot(soln.t*l, ham)

        max_error = np.max(np.abs(np.apply_along_axis(E,0,soln.y)-E0))

        series_error = float('inf')
        n = 1
        while series_error > tol:
            print(series_error)
            ham = harmonic_series(soln.t, n)
            n += 1
            series_error = np.max(np.abs(ham-x))
        
        print(f"Required {n}")

        # plt.plot(soln.t*l, np.apply_along_axis(E,0,soln.y)-E0, c, label=f"$\\lambda={l}$", alpha=0.8)
        plt.plot(soln.t*l, x, c, label=f"$\\lambda={l}$", alpha=0.8)

    plt.legend()
    plt.show()