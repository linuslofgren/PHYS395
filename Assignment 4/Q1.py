import numpy as np

def f(x):
    return x**3 - x + (1/4)

def df(x):
    return 3*x**2 - 1

def find_roots(f, df, search_space):
    zeros = set()

    for start in search_space:
        x = start

        steps = 100

        for _ in range(steps):
            x -= f(x)/df(x)

        if np.abs(f(x)) == 0 and x not in zeros:
            zeros.add(x)

    return zeros

if __name__ == "__main__":
    roots = find_roots(f, df, np.linspace(-10, 10, 100))
    print("Roots:")
    print("f(x)\tx")
    for x in roots:
        print(f"{f(x):.4f}\t{x}")