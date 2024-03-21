from Q1 import evolution
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

E, f = evolution(1.0)

tmax = 10
n = 1024
t = np.linspace(0.0, tmax, n)
l = 1.0
state = np.array([l,0.0]); E0 = E(state)
tol = 1e-12

E0 = E(state)
gamma_forth = gamma(1/4)
T = np.pi**(-1/2)*gamma_forth**2
print(T)

def quad_int(x):
    return 2**(-1/2)/np.sqrt(E0-x**4/4.0)

period, error = quad(quad_int, -1.0, 1.0, epsabs=1e-10)

# period = np.max(soln.t)*2
print(period*2)
plt.show()
