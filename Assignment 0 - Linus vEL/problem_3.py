import numpy as np
import matplotlib.pyplot as plt

def run():
    x = np.linspace(-1, 1, 100)
    polynomials = np.polynomial.legendre.legvander(x, 9)
    _, ax = plt.subplots()
    ax.plot(x, polynomials)
    plt.savefig("output/problem_3.pdf")

if __name__ == "__main__":
    run()