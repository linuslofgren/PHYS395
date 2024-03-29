import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 10-th order Butcher tableau (in quad precision)
A5 = np.array([
     0.5923172126404727187856601017997934066E-1,
    -1.9570364359076037492643214050884060018E-2,
     1.1254400818642955552716244215090748773E-2,
    -0.5593793660812184876817721964475928216E-2,
     1.5881129678659985393652424705934162371E-3,
     1.2815100567004528349616684832951382219E-1,
     1.1965716762484161701032287870890954823E-1,
    -2.4592114619642200389318251686004016630E-2,
     1.0318280670683357408953945056355839486E-2,
    -2.7689943987696030442826307588795957613E-3,
     1.1377628800422460252874127381536557686E-1,
     2.6000465168064151859240589518757397939E-1,
     1.4222222222222222222222222222222222222E-1,
    -2.0690316430958284571760137769754882933E-2,
     4.6871545238699412283907465445931044619E-3,
     1.2123243692686414680141465111883827708E-1,
     2.2899605457899987661169181236146325697E-1,
     3.0903655906408664483376269613044846112E-1,
     1.1965716762484161701032287870890954823E-1,
    -0.9687563141950739739034827969555140871E-2,
     1.1687532956022854521776677788936526508E-1,
     2.4490812891049541889746347938229502468E-1,
     2.7319004362580148889172820022935369566E-1,
     2.5888469960875927151328897146870315648E-1,
     0.5923172126404727187856601017997934066E-1
]).reshape([5,5])

B5 = np.array([
     1.1846344252809454375713202035995868132E-1,
     2.3931433524968323402064575741781909646E-1,
     2.8444444444444444444444444444444444444E-1,
     2.3931433524968323402064575741781909646E-1,
     1.1846344252809454375713202035995868132E-1
])

# 10-th order Gauss-Legendre step
def gl10(f, y, dt):
    n = y.size; g = np.zeros([5,n])
    for k in range(0,16):
        g = np.matmul(A5, g)
        for i in range(0,5):
            g[i] = f(y + g[i]*dt)
    return y + np.dot(B5, g)*dt


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



def create_animatable_pendulum(dt, t, f_ax, f_prime_ax, state, clr):
    
    E0 = E(state)

    pendulum1, = f_ax.plot([0,np.sin(state[0])], [0,-np.cos(state[0])], "o-", color=clr, linewidth=7, ms=15, alpha=0.4)
    pendulum2, = f_ax.plot([np.sin(state[0]),np.sin(state[0])+np.sin(state[1])], [-np.cos(state[0]),-np.cos(state[0])-np.cos(state[1])], "o-", color=clr, linewidth=7, ms=15, alpha=0.4)

    p1_data = np.array([[[0,np.sin(state[0])],[0,-np.cos(state[0])]]])
    p2_data = np.array([[[np.sin(state[0]),np.sin(state[0])+np.sin(state[1])], [-np.cos(state[0]),-np.cos(state[0])-np.cos(state[1])]]])

    end_state_data = np.array([[np.sin(state[0])+np.sin(state[1]), -np.cos(state[0])-np.cos(state[1])]])
    end_state, = f_ax.plot(end_state_data[:,0], end_state_data[:,1], color=clr, linewidth=1, alpha=0.8)

    energy_plt, = f_prime_ax.plot([], color=clr)
    f_ax.set_aspect('equal')

    energy_deltas = []
    
    def step():
        nonlocal state, end_state_data, p1_data, p2_data
        state = gl10(f, state, dt);
        
        theta1 = state[0];
        theta2 = state[1]
        p1_data = np.vstack((p1_data,[ [[0,np.sin(theta1)], [0,-np.cos(theta1)]]]))
        p2_data = np.vstack((p2_data, [[[np.sin(theta1),np.sin(theta1)+np.sin(theta2)], [-np.cos(theta1),-np.cos(theta1)-np.cos(theta2)]]]))
        end_state_data = np.vstack((end_state_data, [np.sin(theta1)+np.sin(theta2), -np.cos(theta1)-np.cos(theta2)]))

        energy_deltas.append(E(state)-E0)

    def show(i):
        nonlocal end_state_data, p1_data, p2_data
        p1_x, p1_y = p1_data[i]
        p2_x, p2_y = p2_data[i]
        pendulum1.set_data(p1_x, p1_y)
        pendulum2.set_data(p2_x, p2_y)
        end_state.set_data(end_state_data[:i,0], end_state_data[:i,1])
        x = np.arange(len(energy_deltas[:i]))*dt
        energy_plt.set_data(x, energy_deltas[:i])
        if len(x) != 0:
            min_ed = np.min(energy_deltas[:i])
            max_ed = np.max(energy_deltas[:i])
            if min_ed != max_ed:
                f_prime_ax.set_xlim(0, np.max(x))
                f_prime_ax.set_ylim(min_ed, max_ed)
        

    return step, show

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

