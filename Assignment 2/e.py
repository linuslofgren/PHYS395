import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
data = np.loadtxt("periodic.dat")

x, y_train = data.T

def basis(x, k_max):
    return np.array([np.cos(2*np.pi*k*x) for k in range(k_max)])

k = 3

A = np.zeros([k, k])
y = np.zeros([k])


fig, ax = plt.subplots()

line = ax.plot(x, y_train)[0]


def update(frame):
    b = basis(x[frame], k)
    global A
    global y
    A += np.outer(b, b)
    y += b*y_train[frame]

    U, S, Vh = np.linalg.svd(A)

    U, S, Vh = np.matrix(U), S, np.matrix(Vh)

    A_inv = Vh.H @ np.diag([0 if np.isclose(0, s, atol=1e-1) else 1/s for s in S]) @ U.H
    c = y @ A_inv
    y_prediction = np.dot(c, basis(x, k)).T

    # x = x

    line.set_xdata(x)
    line.set_ydata(y_prediction)

    return line
    


# for x_i, y_i in zip(x, y_train):
#     b = basis(x_i, k)
#     A += np.outer(b, b)
#     y += b*y_i
    



plt.plot(x, y_train)
# plt.plot(x, y_prediction)
a = animation.FuncAnimation(fig=fig, func=update, frames=len(x))
# plt.plot(x, basis(x, k).T)
plt.show()
