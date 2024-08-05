import numpy as np


def rosenbrock(x):
    dim = len(x)
    fx = np.sum(
        100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
    )
    return fx


# Example usage
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Example input
fx = rosenbrock(x)
print(fx)
