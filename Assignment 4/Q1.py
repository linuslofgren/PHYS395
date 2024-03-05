import numpy as np

def f(x):
    return x**3 - x + (1/4)

def df(x):
    return 3*x**2 - 1

zeros = set()

for start in np.linspace(-100, 100, 100):
    x = start

    N = 10

    for n in range(N):
        x -= f(x)/df(x)

    if np.abs(f(x)) == 0 and x not in zeros:
        zeros.add(x)

print("Roots:")
print("x\t f(x)")
for x in zeros:
    print(x, f(x))