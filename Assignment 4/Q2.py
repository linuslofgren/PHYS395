def bisect(f, a, c):
    """
    a, b, c & x are either:

    _____________________
    |   |       |       |
    |   |       |       |
    a   b      [x]      c

    or:

    _____________________
    |       |       |   |
    |       |       |   |
    a      [x]      b   c
    """

    phi = (5**(1/2)-1)/2

    b = a + (c-a)*phi
    epsilon = 1e-10

    width = lambda: c - a
    left_width = lambda: b - a
    right_width = lambda: c - b
    divide = lambda x, y: x + (y - x)*phi
    squeezed_between = lambda x, y, z: f(y) < f(x) and f(y) < f(z)
    
    while width() > epsilon:
        if left_width() > right_width():
            x = divide(a, b)
            a, b, c = (a, x, b) if squeezed_between(a, x, b) else (x, b, c)
        else:
            x = divide(b, c)
            a, b, c = (a, b, x) if squeezed_between(a, b, x) else (b, x, c)

    return b

# print(bisect(lambda x: (x-2)**2, -5, 5))