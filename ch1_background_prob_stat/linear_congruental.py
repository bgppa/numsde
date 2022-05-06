import numpy as np
import time


def pseudo(x, a, b, c):
        return (a * x + b) % c

def pseudo_sequence(a, b, c, seed = 1, N = 15):
        l = [pseudo(seed, a, b, c)]
        for i in range(1, N):
                l.append(pseudo(l[i-1], a, b, c))
        return l

def find_period(a, b, c, seed = 1):
        l = [pseudo(seed, a, b, c)]
        counter = 0
        repeated = False
        while not repeated:
                l.append(pseudo(l[-1], a, b, c))
                if l[-1] in l[:-1]:
                        repeated = True
                        print(f"Period found: {counter}")
                else:
                        counter += 1
        return counter
                       
def find_period_numpy():
        l = [np.random.randint(2**32)]
        counter = 0
        repeated = False
        while not repeated:
                l.append(np.random.randint(2**32))
                if l[-1] in l[:-1]:
                        repeated = True
                        print(f"Period found: {counter}")
                else:
                        counter += 1
                        if (counter % 1000 == 0):
                                print(f"...counter at {counter}")
        return counter
 

if __name__ == '__main__':
        example = pseudo_sequence(1229, 1, 2048)
        print(example)
        print("Let's find the numpy period and see how fast is reached")
        numpy_int_period = find_period_numpy()
        print("Period found! Let's measure the time")
        now = time.time()
        for i in range(numpy_int_period):
                waste = np.random.randint(2**32)
        print(f"Time elapsed: {time.time() - now} seconds")
