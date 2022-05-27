'''
A simple simulation to check the value of Ito integral.
This is to compare the it
'''
import sys
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    n_steps = 1000
    if len(sys.argv) == 2:
        n_steps = int(sys.argv[1])

    print("Numerical comparison of the Ito, Stratonowitch "
            "and Stieltjes integration")
    # Simulate a Wiener process, w_proc
    w_proc = simulate_wiener(n_steps)

    naive_val = 0.5 * w_proc[n_steps - 1]**2.
    appx_val = 0.
    stratono_val = 0.
    for nth in range(n_steps - 1):
        appx_val += w_proc[nth] * (w_proc[nth + 1] - w_proc[nth])
        stratono_val += (w_proc[nth+1]+w_proc[nth])* (w_proc[nth+1]-w_proc[nth])
    stratono_val /= 2.

    print("--- Wiener integral on [0,1] --- ")
    print(f"{n_steps} steps")
    print(f"naive chain rule value: {naive_val:.2e}")
    print(f"Ito Approximated: {appx_val:.2e}")
    print(f"Ito-correction term (expected -0.5): {appx_val - naive_val:.2e}")
    print(f"Stratonowitch approx: {stratono_val:.2e}")
    print(f"Straton correction term (expected 0.):"
            f" {stratono_val - naive_val:.2e}")

    print("--- Ordinary Stieljes integral of sin on [0,1] ---")
    t_steps = np.linspace(0., 1., n_steps)
    sin_val = np.sin(t_steps)
    naive_val = 0.5 * sin_val[n_steps - 1]**2
    appx_val = 0.
    for nth in range(n_steps - 1):
        appx_val += sin_val[nth] * (sin_val[nth + 1] - sin_val[nth])

    print(f"{n_steps} steps")
    print(f"Naive value: {naive_val:.2e}")
    print(f"Approximated: {appx_val:.2e}")
    print(f"Ito-correction term (expected 0.): {appx_val - naive_val:.2e}")
