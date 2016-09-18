import lib
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import normal
from scipy.optimize import curve_fit

PI = np.pi
RANGE = np.arange(0, 2*PI, 0.1)

N = 60
M = 40
y = np.sin


def nonlin_transform(data, M):
    transf = []

    for m in range(M + 1):
        transf.append([x ** m for x in data])

    return np.matrix(transf)


def lin_model(x, *params):
    return np.dot(params, x)


def polynomial_eval(x, popt):
    return sum([m * x ** list(popt).index(m) for m in popt])


def main():
    data = {x: y(x) + normal(0, 0.1) for x in RANGE}

    x_data = [key for key in data.keys()]
    x_data = lib.n_random(N, x_data)
    y_data = [data[x] for x in x_data]

    popt, pcov = curve_fit(
        f=lin_model,
        xdata=nonlin_transform(x_data, M),
        ydata=y_data,
        p0=np.ones(M+1)
    )

    x_opt = RANGE
    y_opt = [polynomial_eval(x, popt) for x in x_opt]

    plt.plot(RANGE, [y(x) for x in RANGE], 'r')
    plt.plot(x_data, y_data, 'bo')
    plt.plot(x_opt, y_opt, 'g')
    plt.show()

if __name__ == '__main__':
    main()
