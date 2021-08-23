import matplotlib.pyplot as plt
import numpy as np
from numpy import maximum, minimum


class Data:
    def __init__(self, file_name, alfa=1):
        self.file_name = file_name
        self.n = 0
        self.m = 0
        self.x = []
        self.y = []
        self.th = []
        self.mean = []
        self.std_dev = []
        self.alfa = alfa
        self.load_from_file()

    def load_from_file(self):
        with open(self.file_name, 'r') as f:
            line = f.readline()
            tokens = line.strip().split(',')
            self.n = len(tokens)
            for i in range(self.n):
                self.th.append(float(1.0))
            while len(line) > 0:
                self.m += 1
                tokens = line.strip().split(',')
                self.x.append(1.0)
                for i in range(len(tokens) - 1):
                    self.x.append(float(tokens[i]))
                self.y.append(float(tokens[-1]))
                line = f.readline()

        self.th = np.array(self.th, dtype=float)
        self.th = np.reshape(self.th, (self.n, 1))
        self.x = np.array(self.x, dtype=float)
        self.y = np.array(self.y, dtype=float)
        self.x = np.reshape(self.x, (self.m, self.n))
        self.y = np.reshape(self.y, (self.m, 1))
        print("Data successfully loaded")

    def feature_normalization(self):
        self.mean = [float(0) for i in range(self.n)]
        self.std_dev = [1.0 for i in range(self.n)]
        max = [-999999.0 for i in range(self.n)]
        min = [999999.0 for i in range(self.n)]
        for row in self.x:
            for i in range(1, self.n):
                self.mean[i] += row[i]
                max[i] = maximum(max[i], row[i])
                min[i] = minimum(min[i], row[i])
        for i in range(1, self.n):
            self.mean[i] /= self.m
            self.std_dev[i] = max[i] - min[i]
            if self.std_dev[i] == 0:
                self.std_dev[i] = float(1)
        for i in range(self.m):
            for j in range(1, self.n):
                # print("Before:", self.x[i][j], end="    ")
                self.x[i][j] -= self.mean[j]
                self.x[i][j] /= self.std_dev[j]
                # print("After:", self.x[i][j])

    def compute_cost(self):
        h = np.matmul(self.x, self.th)
        h = h - self.y
        h = np.square(h)
        j = np.sum(h, dtype=float)
        j /= float(self.m*2)
        return j

    def adjust_parameters(self):
        h = np.matmul(self.x, self.th)
        diff = h - self.y
        for i in range(self.n):
            self.th[i][0] = float(self.th[i][0] - (self.alfa / self.m) *
                                  np.sum(np.multiply(diff, np.array(self.x[:, [i]]))))

    def gradient_descent(self):
        cost = np.array([])
        iterations = np.array([])
        iteration = 1
        diff = 1.0
        new_cost = 0
        while diff:
            old_cost = self.compute_cost()
            self.adjust_parameters()
            new_cost = self.compute_cost()
            cost = np.append(cost, new_cost)
            iterations = np.append(iterations, iteration)
            iteration += 1
            diff = old_cost-new_cost
            # print("diff", diff)
        # print(iteration)
        plt.plot(cost, iterations)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.show()

    def compute_estimation(self, args):
        return np.matmul(np.array(args, dtype=float), self.th)

    def estimate(self):
        print("Your input:")
        x = [1 for i in range(self.n)]
        for i in range(1, self.n):
            x[i] = input("argument " + str(i) + ": ")
        print(self.compute_estimation(x))


def start():
    d = Data("ex1data2.txt")
    d.feature_normalization()
    d.gradient_descent()
    print(d.th)
    print(d.compute_estimation([1, 2104, 3]) - 399900)


start()
