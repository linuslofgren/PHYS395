import numpy as np
import matplotlib.pyplot as plt

def run():
    try:
        x, y = np.loadtxt("output/problem_1.txt").T
    except FileNotFoundError as e:
        print("Did you forget to run 'problem_1.py?'")
        raise e
    
    _, ax = plt.subplots()
    ax.plot(x, y)
    plt.savefig("output/problem_2.pdf")

if __name__ == "__main__":
    run()
