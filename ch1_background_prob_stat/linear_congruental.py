"""
This mini-library just sum-up basic principles concerning
random number generation and related statistical tests.
"""

import numpy as np
import time
import matplotlib.pyplot as plt


def lcg(between_zero_one, x_seed, param_a, param_b, param_c):
    '''
    Just the linear congruental generator
    '''
    curr_status = x_seed
    def rng():
        nonlocal curr_status
        x_next = (param_a * curr_status + param_b) % param_c
        curr_status = x_next
#        print(f"curr_status = {curr_status}")
        if between_zero_one:
            return x_next / float(param_c)
        else:
            return x_next

    return rng
#---


def lagged_fibonacci(between_zero_one = False, param_r = 17, param_s = 5):
    '''
    Random integer generator with period theoretically
    proved to be (2**17 - 1) * 2**31
    '''
    curr_status = list(range(1, 17))
    def rng():
        nonlocal curr_status
        x_next = (curr_status[-param_r+1] + curr_status[-param_s+1]) % 2**31
        curr_status = curr_status[1:] + [x_next]
#        print(f"{curr_status}")
        if between_zero_one:
            return x_next / float(2 ** 31)
        else:
            return x_next

    return rng
#---


def f_find_period (random_generator):
    """
    Generate up to 2**32-1 random positive integers by using the
    routine given as parameter. Stop when the same integer is generated.
    Show then the next 10 numbers in the sequence, just as a check of
    the repetition.
    """
    pseudo_sequence = [random_generator()]
    counter = 0
    repeated = False
    while not repeated and counter < 2**32:
        # Add a random number to the sequence and increase the period counter
        pseudo_sequence.append(random_generator())
        counter += 1
        if counter % 1000 == 0:
            print(f"...[{counter}]")

        # Check if the newly added number was already present
        for idx, elm in enumerate(pseudo_sequence[:-1]):
            if pseudo_sequence[-1] == elm:
                print(f"found {pseudo_sequence[-1]} = {elm}")
                # if the integer has ALREADY been generated...
                # ...generate 10 new random integers
                for tmp in range(9):
                    pseudo_sequence.append(random_generator())
                # print the two sequences
                print(f"First seq: {pseudo_sequence[idx:idx+10]}")
                print(f"Secnd seq: {pseudo_sequence[-10:]}")
                if pseudo_sequence[idx:idx+10] == pseudo_sequence[-10:]:
                    repeated = True
                    break

    print(f"Period: {counter}")
    return counter
#---


def stat_analysis(random_generator, n_samples = 100_000, bins = 50):
    """
    Simple statistical analysis of a uniform random generator.
    More checks will be added in the future.
    """
    samples = np.zeros(n_samples)
    for tmp in range(n_samples):
        samples[tmp] = random_generator()

    mean = np.mean(samples)
    var = np.std(samples) ** 2
    print(f"Mean:\t\t\t\t{mean:.3f}\t[compare to {1. / 2: .3f}]")
    print(f"Variance:\t\t\t{var:.3f}\t[compare to {1. / 12 : .3f}]")

    # An easy histogram analysis
    step_size = 1. / bins
    frequencies = np.zeros(bins)

    # For each number sampled, find the interval to which it belongs
    for num in samples:
        for nth_bin in range(bins):
            if nth_bin * step_size <= num and num <= (nth_bin+1) * step_size:
                hist_index = nth_bin
        frequencies[hist_index] += 1

    # Print the frequencies in a percentage format
    print("-"*13 + " Histogram " + "-" * 12)
    frequencies = frequencies / n_samples * 100.
    for tmp in range(bins):
        print(f"bin({tmp + 1})\t[{tmp*step_size:.2f},{(tmp+1)*step_size:.2f}]"
                f"\t:\t{frequencies[tmp]:.2f}") 
    print("-"*36)


    return 0
#---


if __name__ == '__main__':
    gen = lcg(True, 0, 1229, 1, 2048)
#    f_find_period(gen)
#    numpy_gen = np.random.randint(2**32)
    gen2 = lagged_fibonacci(True)
#    print(f"Expected period: {(2 ** 17 -1) * 2 ** 31}")
#    f_find_period(gen2)
    
#    stat_analysis(np.random.uniform)
    print("Analysis of the LGC")
    stat_analysis(gen)
    print("Analysis of Fibonacci")
    stat_analysis(gen2)
    print("Analysis of Numpy")
    stat_analysis(np.random.uniform)
