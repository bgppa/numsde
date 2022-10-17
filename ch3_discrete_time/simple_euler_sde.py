'''
Just to simply remind the stochastic simulation using Euler
'''
import numpy as np
import matplotlib.pyplot as plt


def a(t, x):
    '''
    Deterministic drift
    '''
    return 2. * x


def b(t, x):
    '''
    Stochastic oscillation
    '''
    return 3.


if __name__ == '__main__':
    # Number of parts in which we subdivide the unitary time interval, e.g. 2
    ntimediv = 20
    # Number of final discretization points; if divide in 2, obtain 3
    N = ntimediv + 1
    # time step
    h = 1. / ntimediv
#    xaxis = np.linspace(0, 1, N)

    y_appx = np.zeros(N)
    x_axis = np.zeros(N)


    y_appx[0] = 1.
    # Here comes the simulation
    for n in range(ntimediv):
        x_axis[n + 1] = h * (n +1)
        y_appx[n+1] = y_appx[n] + a(n * h, y_appx[n]) * h
        y_appx[n+1] += b(n*h, y_appx[n]) * np.random.normal()*np.sqrt(h)


    # Plot the results
    plt.plot(x_axis, y_appx, label="Euler SDE")
    plt.grid()
    plt.legend()
    plt.show()


