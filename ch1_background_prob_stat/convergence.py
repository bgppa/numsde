'''
Exercise 1.4.1 on the convergence of Random Variables.
'''
import numpy as np
import matplotlib.pyplot as plt


def histo (samples, display=True, start=-4., end=4., verbose=True):
    '''
    Display a histogram with the sampled data
    '''
    # The number of bins is selected according to the Sturge's rule
    n_bins = int(1. + 3.322 * np.log(len(samples)))
    freqs = np.zeros(n_bins-1)
    intervals = np.linspace(start, end, n_bins)
    for point in samples:
        # Find the interval in which the point is contained
        for num in range(n_bins - 1):
            if intervals[num] <= point <= intervals[num + 1]:
                freqs[num] += 1
                break

    # Normalize the frequences
    freqs = freqs * 100. / len(samples)

    if verbose:
        print("---- Frequencies ----")
        for num in range(n_bins-1):
            print(f"[{intervals[num]:.2f}, {intervals[num+1]:.2f}]"
                    f"   {freqs[num]:.2f}")
        print(f"Total: {np.sum(freqs):.2f}%")

    if display:
        plt.plot(intervals[:-1], freqs)
        plt.grid()
        plt.title("Histogram of frequencies")
        plt.show()

    return freqs
#---


if __name__ == '__main__':
    # Analysis of the convergence for the random variables decribed in the book

    # We'll study the error for sample sizes going from 2**5 to 2**9
    max_N = 10
    l2errors = np.zeros(max_N - 1 - 5)

    for n in range(5, max_N):
        X_rv = np.random.uniform(0., 1., 2**n)
        Zn_rv = np.random.normal(0., 1./n, 2**n)
        Yn_rv = X_rv + Zn_rv

        print(f"Histogram for n = {n}")
        histo(Yn_rv, True, -3, 3, False)
        l2errors[n - 1 - 5] = np.mean((Yn_rv - X_rv)**2)

    plt.plot(list(range(5, max_N - 1)), l2errors)
    plt.title("L2 convergence error for a trivial example")
    plt.xlabel("log_2(sample size)")
    plt.ylabel("l2 error")
    plt.grid()
    plt.show()
