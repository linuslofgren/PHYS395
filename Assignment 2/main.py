import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("periodic.dat")

x, y_training = data.T

def basis(x, k_max):
    return np.array([np.cos(2*np.pi*k*x) for k in range(k_max)])


for k in [3, 7]:
    print(f"Using K_max = {k}")
    A = np.zeros([k, k])
    y = np.zeros([k])
    for x_i, y_i in zip(x, y_training):
        b = basis(x_i, k)
        A += np.outer(b, b)
        y += b*y_i
        

    U, S, Vh = np.linalg.svd(A)
    U, S, Vh = np.matrix(U), S, np.matrix(Vh)

    print(f"Condition number: κ={np.max(S)/np.min(S):.2f}")

    A_inv = Vh.H @ np.diag([0 if np.isclose(0, s, atol=1e-1) else 1/s for s in S]) @ U.H
    c = y @ A_inv
    y_prediction = np.dot(c, basis(x, k)).T

    chi_squared = np.sum(np.square(y_training-y_prediction))
    print(f"Goodness of fit: 𝛘²={chi_squared}")

    fig, ax = plt.subplots()
    plt.title(f"K={k}")
    plt.plot(x, y_training, "+", label="Source data")
    plt.plot(x, y_prediction, label="Fitted model")
    plt.legend()
    plt.savefig(f"plot k={k}.pdf")
