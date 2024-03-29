import numpy as np
from gl import gl10

omega = 1.0

def f(state):
    theta1, theta2, p1, p2 = state
    
    theta1_p = get_theta1_p(state)
    theta2_p = get_theta2_p(state)

    p1_p = -3*omega**2*np.sin(theta1)-np.sin(theta1-theta2)*theta1_p*theta2_p
    p2_p = -omega**2*np.sin(theta2)+np.sin(theta1-theta2)*theta1_p*theta2_p

    return np.array([theta1_p, theta2_p, p1_p, p2_p])

def get_theta1_p(state):
    theta1, theta2, p1, p2 = state
    return (2/3*p1-np.cos(theta1-theta2)*p2)/(16/9-np.power(np.cos(theta1 - theta2), 2))

def get_theta2_p(state):
    theta1, theta2, p1, p2 = state
    return (8/3*p2-np.cos(theta1-theta2)*p1)/(16/9-np.power(np.cos(theta1 - theta2), 2))

def E(state):
    theta1, theta2, p1, p2 = state

    theta1_p = get_theta1_p(state)
    theta2_p = get_theta2_p(state)


    energy = 4/3*theta1_p**2+1/3*theta2_p**2+theta1_p*theta2_p*np.cos(theta1-theta2)-omega**2*(3*np.cos(theta1) + np.cos(theta2))
    return energy


if __name__ == "__main__":
    dt = 0.02
    t = 10/omega
    iterations = int(t/dt)

    passed = 0
    fail = 0

    n = 10
    for t1 in np.linspace(-np.pi, np.pi, n):
        for t2 in np.linspace(-np.pi, np.pi, n):
            print(f"{(passed+fail)/(n*n):.2%}")
            state = np.array([t1, t2, 0, 0])
            E0 = E(state)

            max_error = 0

            for _ in range(iterations):
                state = gl10(f, state, dt);
                error = np.abs(E(state)-E0)

                max_error = error if error > max_error else max_error
                if error > 10e-12:
                    print("❌ Error too large...")
                    fail += 1
                    break
            else:
                print(f"✅ Error within limits with time step {dt:.2f}")
                passed += 1
    
    print("#"*10)
    if fail > 0:
        print(f"{error} initial conditions failed with dt={dt}")
    else:
        print(f"All initial conditions had accuracy < 10E-12 with dt={dt}")