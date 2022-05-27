'''
This is my first naive experiment in solving an sde using
a form of signature scheme.
In this specific example we work with a Geometric Brownian Motion
'''

import torch
import signatory as sg
import numpy as np
import matplotlib.pyplot as plt


def parametrized(original_twod_curve):
    pass
    '''
    Sig[nh, (n+1)h](X_t) = Sig[0,1](X_parametrized(s))
    '''

def mysigcoeff (two_dim_path):
    '''
    Given a path as tensor of shape (1, no_matter_number, 2),
    returns some specific coefficient of its signature transform.
    When the path is (t, W_t), these terms correspond to
    the Numerical scheme 3.23 at page 81.
    Memento: when computing the integration HERE, the domain is
    assumed to be the full [0,1].
    Therefore you should reparametrize the path before using it as argument
    here, so that sig[0,1](new_path) = sig[h, h+dt](original_path).
    '''

    tmp = sg.signature(two_dim_path, 2, scalar_term = True)
    return tmp[0][0], tmp[0][1], tmp[0][2], tmp[0][6]
#---


def v_coeff(x_n):
    '''
    x_(n+1) = v_coeff(x_n) \dot mysigcoeff[nh, (n+1)h]
    '''
    return [x_n, (a - 0.5 * b**2) * x_n, b * x_n, b**2 * x_n]
#---


#### - Section on ordinary simulation for a gbm

#DONE
def simulate_wiener (n_steps, makeplot = False):
    '''
    Random-Walk based simulation of a Wiener process
    '''
    # The various variables +-1 for the random walk, starting from 0
    rw_samples=[1 if np.random.uniform()>0.5 else -1 for n in range(n_steps-1)]
    rw_samples = np.asarray(rw_samples)

    # Wiener process variable and time step
    wiener = np.zeros(n_steps)
    delta = np.sqrt(1. / (n_steps - 1))

    # The value corresponds to the sum of the random walks, rescaled
    for nth in range(n_steps):
        wiener[nth] = np.sum(rw_samples[:nth]) * delta

    if makeplot:
        plt.plot(np.linspace(0., 1., n_steps), wiener, label = 'rw')
        plt.legend()
        plt.title("Wiener simulated by discrete RW")
        plt.grid()
        plt.show()

    return wiener
#---

#DONE
def simulate_gbm(coeff_a, coeff_b, n_steps, makeplot = False):
    '''
    True solution for a Geometric Brownian Motion
    '''
    underlying_wiener = simulate_wiener(n_steps)
    res = np.zeros(n_steps)
    delta_t = 1. / (n_steps - 1)
    for nth in range(n_steps):
        res[nth] = np.exp((coeff_a - 0.5 * (coeff_b**2)) * nth * delta_t +
            coeff_b * underlying_wiener[nth])
    return res
#---


if __name__ == '__main__':
    tmp_n = 500
    result = simulate_gbm(0.3, 0.2, tmp_n)
    plt.plot(np.linspace(0, 1, tmp_n), result)
    plt.grid()
    plt.show()
            
