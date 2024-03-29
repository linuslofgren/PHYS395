import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Q2 import create_animatable_pendulum
from Q1 import omega

if __name__ == "__main__":
    fig, (f_ax, f_prime_ax) = plt.subplots(2, 1, figsize=(14, 8))
    f_ax.add_patch(plt.Circle((0, 0), 2.0, color='lightgray', fill=False))

    dt = 0.02
    t = 100/omega
    frames = int(t/dt)

    offset = 10e-6

    state1 = np.array([2*np.pi/3,          2*np.pi/3,         0.0, 0.0]) # theta1, theta2, p1, p2
    state2 = np.array([2*np.pi/3-offset,   2*np.pi/3-offset,  0.0, 0.0]) # theta1, theta2, p1, p2
    state3 = np.array([2*np.pi/3+offset,   2*np.pi/3+offset,  0.0, 0.0]) # theta1, theta2, p1, p2

    st,  show  = create_animatable_pendulum(dt, t, f_ax, f_prime_ax, state1, "lightgreen")
    st2, show2 = create_animatable_pendulum(dt, t, f_ax, f_prime_ax, state2, "orange")
    st3, show3 = create_animatable_pendulum(dt, t, f_ax, f_prime_ax, state3, "lightblue")

    def animate(i):
        show(i)
        show2(i)
        show3(i)


    print("Calculating...")
    for i in range(frames):
        if(i%100==0):
            print(f"{i/frames:.2%}")
        st()
        st2()
        st3()

    print(f"Animating {frames} frames")
    animation = animation.FuncAnimation(fig, animate, frames=frames, interval=10, repeat=False)
    plt.show()
    # animation.save('Q1-3.mp4')

