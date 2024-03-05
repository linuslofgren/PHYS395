from Q2 import bisect


def f(x):
    return (x**2-1)**2+x

print(f(bisect(f, -100, 100)))