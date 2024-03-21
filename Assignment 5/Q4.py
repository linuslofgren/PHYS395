from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

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
    x, v = soln.y
    # plt.plot(soln.t*l, x, label=f"$\\lambda={l}$", alpha=0.8)
    # plt.plot(soln.t, x, label=theta)
    period = np.max(soln.t)*2
    return period

thetas = np.linspace(0, 0.99*np.pi, 40)[1:]

periods = np.vectorize(period_for)(thetas)
print(periods)
plt.plot(thetas, periods)
# plt.legend()

plt.show()