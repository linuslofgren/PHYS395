from Q1 import evolution
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

E, f = evolution(1.0)

tmax = 10
n = 1024
t = np.linspace(0.0, tmax, n)
l = 1.0
state = np.array([l,0.0]); E0 = E(state)
tol = 1e-12


def stop(t, state):
    x, v = state
    return v

stop.terminal = True
stop.direction = 1

soln = solve_ivp(f, [t[0],t[-1]], state, method='Radau', t_eval=t, atol=1e-12, events=stop)
x, v = soln.y
plt.plot(soln.t*l, x, label=f"$\\lambda={l}$", alpha=0.8)

period = np.max(soln.t)*2
print(period)
plt.show()
