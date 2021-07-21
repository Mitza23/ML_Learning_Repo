from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def read_file(file_name):
    x = []
    y = []
    f = open(file_name, 'r')
    line = f.readline()
    while len(line) > 0:
        tokens = line.strip().split(",")
        x.append(float(tokens[0]))
        y.append(float(tokens[1]))
        line = f.readline()
    return x, y


def plot_data(x, y):
    xpoints = np.array(x)
    ypoints = np.array(y)

    plt.plot(xpoints, ypoints, 'o')
    plt.show()


def compute_h(x, th0, th1):
    h = np.array(x)
    h *= th1
    h += th0
    return h


def compute_cost(x, th0, th1, y):
    h = compute_h(x, th0, th1)
    result = h - y
    result = np.square(result)
    j = np.sum(result, dtype=float)
    j /= (len(x)*2)
    return j


def adjust_parameters(x, th0, th1, y, alfa):
    h = compute_h(x, th0, th1)
    diff = h - y
    new_th0 = th0 - (alfa / len(x)) * np.sum(diff, dtype=float)
    new_th1 = th1 - (alfa / len(x)) * np.sum(np.multiply(diff, x), dtype=float)
    return new_th0, new_th1


def plot(p0, p1, cost):
    p0 = np.array(p0)
    p1 = np.array(p1)
    cost = np.array(cost)

    X, Y = np.meshgrid(p0, p1)
    Z = cost.reshape(cost, (len(cost), 1))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def gradient_descent(x, th0, th1, y, alfa):
    cost = []
    p0 = []
    p1 = []
    new_th0, new_th1 = adjust_parameters(x, th0, th1, y, alfa)
    while abs(th0 - new_th0) > 0.001 or abs(th1 - new_th1) > 0.001:
    # for i in range(10):
    #     print(abs(th0 - new_th0), abs(th1 - new_th1))
        c = compute_cost(x, th0, th1, y)
        cost.append(c)
        print("cost:", c)
        p0.append(th0)
        p1.append(th1)
        th0 = deepcopy(new_th0)
        th1 = deepcopy(new_th1)
        new_th0, new_th1 = adjust_parameters(x, th0, th1, y, alfa)

    # plot(p0, p1, cost)
    return th0, th1


def run():
    x, y = read_file("ex1data1.txt")
    gradient_descent(x, 1.0, 1.0, y, 0.001)


if __name__ == '__main__':
    run()
