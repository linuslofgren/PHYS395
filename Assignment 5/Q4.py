from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from Q1 import unit_period

def evolution(l=1.0):
    omega = 1.0
    def E(state):
        theta, d_theta = state
        return d_theta**2/2-omega**2*np.cos(theta)

    def f(t, state):
        theta, d_theta = state
        return np.array([d_theta, -omega**2*np.sin(theta)])
    
    return E, f

tmax = 10
n = 1024
t = np.linspace(0.0, tmax, n)
tol = 1e-12

periods = np.zeros([100])

def period_for(theta):
    E, f = evolution(theta)

    # l = 1.0
    state = np.array([theta,0.0]); E0 = E(state)


    def stop(t, state):
        x, v = state
        return v

    stop.terminal = True
    stop.direction = 1

    soln = solve_ivp(f, [t[0],t[-1]], state, method='Radau', t_eval=t, atol=1e-12, events=stop)

    period = np.max(soln.t)*2
    return period

thetas = np.linspace(0, 0.99*np.pi, 40)[1:]

periods = np.vectorize(period_for)(thetas)

fig, ax = plt.subplots()

for t, p in zip(thetas, periods):
    if np.abs(np.abs(p-unit_period())/unit_period()-0.1) < 0.01:
        ax.axvline(x=t, c="y")
ax.axhline(y=unit_period(), color="r", label="Harmonic approximation")
ax.set_title("Period as a function of $\\Theta$.\n Yellow indicated amplitudes which gives a 10% difference from of harmonic approximation")
ax.set_xlabel("$\\Theta$")
ax.set_ylabel("Period")
ax.plot(thetas, periods)
ax.legend()
fig.savefig("Q4.pdf")
plt.show()