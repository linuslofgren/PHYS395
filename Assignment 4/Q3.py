from Q2 import bisect
import numpy as np

def f(x):
    return (x**2-1)**2+x
# Move a window of width 2 in integer steps from -10 to 10 to find all minima
# will fail if there are more than two minima within a distance of 2
minima = set()
for a in np.linspace(-10, 10, 20):
    candidate = bisect(f, a, a+2)
    if np.abs(candidate-a) > 0.1 and np.abs(candidate-a-2) > 0.1:
        minima.add(f"{candidate:.4f}")

print("Minima:")
print("x\tf(x)")
for x in minima:
    print(f"{x}\t{f(float(x)):.4f}")