from Q1 import evolution, unit_period
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def integrate_period(l=1.0):
    E, f = evolution(1.0)

    tmax = 10
    n = 1024
    t = np.linspace(0.0, tmax, n)
    
    state = np.array([l,0.0]); E0 = E(state)
    tol = 1e-12


    def stop(t, state):
        x, v = state
        return v

    stop.terminal = True
    stop.direction = 1

    soln = solve_ivp(f, [t[0],t[-1]], state, method='Radau', t_eval=t, atol=1e-12, events=stop)
    x, v = soln.y

    return x, v, soln.t

if __name__ == "__main__":
    l = 1.0

    x, v, t = integrate_period(l)

    fig, ax = plt.subplots()
    ax.plot(t*l, x, label=f"$\\lambda={l}$", alpha=0.8)

    period = np.max(t)*2
    print("Period with solve_ivp:", period)
    print(f"Relative error: {(period-unit_period())/unit_period():.4%}")
    fig.savefig("Q2.pdf")
    plt.show()
