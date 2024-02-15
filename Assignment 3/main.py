# Linus von Ekensteen Löfgren
# Thanks to Michael Williams & Iman Fortin for comments

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("periodic.dat")

x, y_training = data.T

def basis(x, k_max):
    return np.array([np.cos(2*np.pi*k*x) for k in range(k_max)])


for k in [3, 7]:
    print("#"*4 + f" Using K_max = {k} " + "#"*4)
    A = np.zeros([k, k])
    y = np.zeros([k])

    b = basis(x, k)

    A = b @ b.T
    y = y_training @ b.T 

    U, S, Vh = np.linalg.svd(A)

    print(f"Condition number: κ={np.max(S)/np.min(S):.2f}")

    # Round singular values close to zero to zero
    S = np.diag(np.divide(np.ones_like(S), S, out=np.zeros_like(S), where=~np.isclose(0, S, atol=1e-1)))

    A_inv = Vh.T @ S @ U.T
    coefficients = y @ A_inv
    y_prediction = (coefficients @ basis(x, k)).T

    chi_squared = np.sum(np.square(y_training-y_prediction.T))

    # Degrees of freedom = nbr of observations - nbr of params
    dof = len(x)-k

    print(f"Goodness of fit: 𝛘²={chi_squared:.2f}, reduced 𝛘²={chi_squared/dof:.2f}")

    fig, ax = plt.subplots()
    plt.title(f"Fitted with K={k} parameters")
    plt.plot(x, y_training, "+", label="Source data")
    plt.plot(x, y_prediction, label="Fitted model")
    plt.legend()
    plt.savefig(f"plot k={k}.pdf")
