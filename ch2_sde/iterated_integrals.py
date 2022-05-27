'''
Comparing the interated integral computed by using the
Signatory library, and the one suggested on the book when using
the KL expansion.
The path is a timed-brownian Motion: t -> (t, W_t)
'''
import torch
import numpy as np
import signatory as sg
import matplotlib.pyplot as plt


# Yes, the function below is taken from the other file, but for now is a bit
# soon to really re-arrange everything into a library.
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


def f_epsilon(extended_path, component, delta):
    '''
    returns W^j_delta / sqrt(delta). W is the generic extended path, such that
    for j = 1 we have the Wiener simulation, for j = 0 just the time t.
    '''
    return extended_path[-1][component] / np.sqrt(delta)


def f_acoeff(extended_path, component, delta, erre):
    '''
    formula (3.25) page 82
    '''
    res = 1.

    res = 1. #the big integral, TO DO

    return res * 2. / delta
#---
   

def biga():
    return 1.
    

def iterated_integrals(timed_path, delta = 1., p_series = 50):
    '''
    Using the book
    '''
    # Auxiliary variables
    delta = torch.tensor(delta)

    # actually a value to better compute
    a10 = f_acoeff(timed_path, 1, delta, 0)

    eps1 = f_epsilon(timed_path, 1, delta)

    jp0 = delta
    jp1 = torch.sqrt(delta) * eps1
    jp00 = 0.5 * (delta ** 2)

    # jp10 TO COMPLETE
    jp10 = 0.5 * delta * (jp1 + a10)
    # ---

    jp01 = delta * jp1 - jp10
    jp11 = 0.5 * (jp1) ** 2

    jp000 = ((delta)**3) / 6.

    jp001 = 0.

    jp010 = 0.
    jp011 = 0.
    jp100 = jp001 + 0.5 * (delta**2) * a10
    jp101 = 0.
    jp110 = delta * jp11 - 
    jp111 = ((jp1) ** 3) / 6.

    return torch.tensor([jp0, jp1, jp00, jp01, jp10, jp11,
                        jp000, jp001, jp010, jp011,
                        jp100, jp101, jp110, jp111])
#---



if __name__ == '__main__':
    # First of all, simulate a Wiener process
    level = 4
    n_points = 2 ** level
    wproc = simulate_wiener(n_points, False)

    # Converting the path into a torch tensor, so to use Signatory
    the_path = torch.zeros(1, n_points, 2)
    the_path[0][:, 0] = torch.linspace(0, 1, n_points)
    the_path[0][:, 1] = torch.tensor(wproc)

    sig = sg.signature(the_path, 3)
    print(f"With signatory:\n{sig[0]}")

    # Now, I want to compute the iterated integrals according to the KP
    # formalism
    my_sig = iterated_integrals(the_path[0])
    # the_path[0][i][j] is the value at time i of the j-th component
    print(f"With KP method:\n{my_sig}")
