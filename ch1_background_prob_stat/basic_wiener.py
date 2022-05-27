'''
Simple examples just to remark three different ways to simulate a Wiener
Process:
(1) cumulative sums of gaussians
(2) expoiting a limiting random walk [central limit theorem]
(3) by truncating the KL expansion
My favorite is (2) for the convergence in distribution which is cystal
clear.
'''
import numpy as np
import matplotlib.pyplot as plt

def path(n_steps = 10):
    '''
    Simulation using cumulative sums of gaussians
    '''
    # Wiener array and time step
    wiener = np.zeros(n_steps)
    delta = np.sqrt(1. / (n_steps - 1))

    # Each Wiener value is simply obtained by summing a new gaussian
    for nth_step in range(1, n_steps):
        wiener[nth_step] = wiener[nth_step - 1] + delta * np.random.normal()

    plt.plot(np.linspace(0., 1., n_steps), wiener, label = 'gss')
    plt.title("Wiener simulated by cumulated Gaussians")
    plt.legend()
    plt.grid()
#    plt.show()
    return wiener
#---


def rw(n_steps = 10):
    '''
    Simulation using random walk discretization
    '''
    # The various variables +-1 for the random walk, starting from 0
    rw_samples=[1 if np.random.uniform()>0.5 else -1 for n in range(n_steps-1)]
    rw_samples = np.asarray(rw_samples)
#    print(rw_samples)

    # Wiener process variable and time step
    wiener = np.zeros(n_steps)
    delta = np.sqrt(1. / (n_steps - 1))

    # The value corresponds to the sum of the random walks, rescaled
    for nth in range(n_steps):
        wiener[nth] = np.sum(rw_samples[:nth]) * delta

    plt.plot(np.linspace(0., 1., n_steps), wiener, label = 'rw')
    plt.legend()
    plt.title("Wiener simulated by discrete RW")
    plt.grid()
#    plt.show()
    return wiener


def kl_expansion(truncation = 50, n_steps = 10):
    '''
    Simulating on the interval [0,1], n_steps time steps
    using the Karhunen-Loeve Expansion
    '''
    # Store _truncation_ gaussian coefficients
    coeff_z = np.random.normal(0., 1., truncation)

    # Function to compute phi_n(t), given n and t
    def phi_n (param_n, time_t):
        '''
        Evaluate the single value of phi_n(t)
        '''
        sin_part = np.sin((2*param_n +1) * np.pi * time_t / 2.)
        coeff = np.sqrt(2) * 2. / ((2. * param_n + 1) * np.pi)
        return coeff * sin_part

    # Produce the complete collections of phi_n for a fixed given t
    def coeff_phi(time_t):
        '''
        Return _truncation_ phi functions evaluated at time t
        '''
        phis = np.zeros(truncation)
        for nth in range(truncation):
            phis[nth] = phi_n(nth, time_t)
        return phis

    # Wiener array
    wiener = np.zeros(n_steps)
    times = np.linspace(0., 1., n_steps)
    for nth in range(n_steps):
        # Each value is the sum of all the z coeffs and all the phi
        wiener[nth] = np.sum(coeff_z * coeff_phi(times[nth]))

    plt.plot(np.linspace(0., 1., n_steps), wiener, label = 'kl')
    plt.legend()
    plt.title("Wiener simulated by KL expansion")
#    plt.grid()
#    plt.show()
    return wiener


if __name__ == '__main__':
#    path(n_steps = 10_000)
#    rw(n_steps = 10_000)
    kl_expansion(truncation = 5, n_steps = 10_000)
    plt.grid()
    plt.show()
