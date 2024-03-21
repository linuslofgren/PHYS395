from Q1 import evolution, unit_period
from scipy.integrate import quad
import numpy as np
from Q2 import integrate_period

E, f = evolution(1.0)

l = 1.0
state = np.array([l,0.0]);
E0 = E(state)

def quad_int(x):
    return 2**(-1/2)/np.sqrt(E0-x**4/4.0)

half_period, error = quad(quad_int, -1.0, 1.0, epsabs=1e-16)
quad_period = 2*half_period

ix, iv, it = integrate_period()

integrated_period = np.max(it)*2

print(f"Quad error: {(quad_period-unit_period())/unit_period():.8%} ({(quad_period-unit_period())/unit_period()})")
print(f"Integration error: {(integrated_period-unit_period())/unit_period():.4%}")
print("Quadrature performs orders of magnitude better but cannot reach 10e-16 to the analytical expression.")

