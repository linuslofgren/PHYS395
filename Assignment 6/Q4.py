import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from joblib import Parallel, delayed
from gl import gl10
from Q1 import omega, f
n = 10
# theta_1_range = np.linspace(-np.pi+0.01, np.pi-0.01, n)
# theta_2_range = np.linspace(-np.pi+0.01, np.pi-0.01, n)
# theta_1_range = np.linspace(1.55, 2.144, n)
# theta_2_range = np.linspace(-1.2, -0.72, n)
theta_1_range = np.linspace(1.55, 1.72, n)
theta_2_range = np.linspace(-1.2, -0.72, n)
res = np.zeros((n, n))
dt = 0.02
time_limit = 100/omega

def get_steps(initial):
    state = initial
    for i in range(int(time_limit/dt)):
        if np.abs(state[0]) > np.pi or np.abs(state[1]) > np.pi:
            return i*dt
        state = gl10(f, state, dt)
    return 0.0

from itertools import product


r = Parallel(n_jobs=16, verbose=True)(delayed(get_steps)(np.array([t1, t2, 0.0, 0.0])) for t1, t2 in product(theta_1_range, theta_2_range))
res = np.array(r).reshape(n, n)


cmap = cm.afmhot; cmap.set_under('lightgray')

plt.imshow(res-(res==0.0), extent=(theta_1_range[0], theta_1_range[-1], theta_2_range[0], theta_2_range[-1]), vmin=0.0, cmap=cmap, norm='linear', aspect='equal', interpolation='none')
plt.show()
