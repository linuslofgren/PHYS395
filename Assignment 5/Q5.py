from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np


def cheb(v):
    def f(x, state):
        y, y_p = state
        return [y_p, (x*y_p-(v**2)*y)/(1-x**2)]
    
    if v%2 == 0:
        state = np.array([1, 0])
    else:
        state = np.array([0, 1])


    soln = solve_ivp(f, (0.0, 1.0), state, method='Radau', max_step=0.01)
    y, _ = soln.y
    scalar = 1/y[-1]

    f = 1 if v%2 == 0 else -1

    return np.hstack([-np.flip(soln.t), soln.t]), np.hstack([f*np.flip(y*scalar), y*scalar])

fig, ax = plt.subplots()

for i in range(7):
    x, y = cheb(i)

    ax.plot(x, y, label=f"v={i}")

x, y = cheb(3/2)

ax.plot(x, y, label=f"v=3/2")
print("For v=3/2 the curve bends and does not look orthogonal to the others.")

ax.set_title(r"Solutions to Chebyshev equation $(1-x2)y'' -xy' +v^2y =0$")
ax.legend()
fig.savefig("Q5.pdf")
plt.show()
