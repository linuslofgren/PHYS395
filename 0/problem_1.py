import numpy as np

def gaussian(x):
    return np.exp(-np.square(x)/2)

def run():
    x = np.linspace(-5, 5, 100)
    y = gaussian(x)
    np.savetxt("output/problem_1.txt", np.transpose([x, y]), fmt=["%f", "%f"], delimiter="\t")

if __name__ == "__main__":
    run()