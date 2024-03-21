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

def unit_period():
    gamma_forth = gamma(1/4)
    T = np.pi**(-1/2)*gamma_forth**2
    return T

def harmonic_series(t, n):
    n = np.arange(1, n+1)
    T = unit_period()
    return np.sum(2**(1/2)*4*np.pi/T*np.cos(np.outer((n-1/2)*4*np.pi/T, t))/np.cosh((n-1/2)*np.pi)[:,None], axis=0)

if __name__ == "__main__":
    fig, ax = plt.subplots()
    
    # Plot parameter
    spacing = 4
    tmax = 2*unit_period()

    # IVP parameters
    n = 1024
    tol = 1e-12

    for i, l in enumerate([1/2, 1, 2]):

        # Scaling
        t = np.linspace(0.0, tmax/l, n)

        E, f = evolution(1.0)
        
        initial_state = np.array([l,0.0]);

        soln = solve_ivp(
            f,
            [t[0],t[-1]],
            initial_state,
            method='Radau',
            t_eval=t,
            atol=tol
        )
        x, v = soln.y

        ax.plot(soln.t*l, x/l, linestyle=(i*(1+spacing)/3, (1, spacing)), label=f"$\\lambda={l}$", alpha=0.8)

    for order in range(1, 4):
        t = np.linspace(0, tmax, n)
        ham = harmonic_series(t, order)
        ax.plot(t, ham, label=f"$n={order}$", alpha=0.4)

    ax.set_title("Numerical solutions with amplitude $\\lambda$ and harmonic expansion with $n$ terms.\n $n=2$ gives a good solution and $n=3$ is practically perfect.")
    ax.legend()
    fig.savefig("Q1.pdf")
    plt.show()