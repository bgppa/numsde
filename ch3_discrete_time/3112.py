'''
An exercise to study the distribution of the CUMULATED roundoff error.
We assume to work with an algorithm wih N steps (e.g. Euler, here)
Each step produces an output containing a roundoff error.
The sums of all these errors is called the Cumulated Roundoff error.

By definition it is a sum of iid uniform distributions.
Therefore can be approximated by a Gaussian for large values of samples (steps).
It allows to compute a confidence interval where to expect this sum to lie.

We repeat an Euler ODE simulation L times, each time:
 - computing the CRE
 - its mean and variance (empirical)
observing that the actual CRE value oscillates a little bit but it stays
always with basically the same mean and variance predicted from the theory.

Each experiment is repeated again by choosing a different value of N.
'''
import numpy as np
import matplotlib.pyplot as plt

# Force a roundoff to 4 digits
ROUNDOFF = 4



# We'll go for a simple Euler method, and sums the error at every step

def a(x_n, t_n):
   return x_n


def x_sol(t):
    return np.exp(t)

# Number of total simulations
L = 200
powerN = 3


N = pow(2, powerN)
h = 1. / N
tmesh = np.linspace(0, 1, N + 1)

nth_attempt = np.array(range(L))
nth_error = np.zeros(L)
nth_var = np.zeros(L)
nth_means = np.zeros(L)
nth_predicted_var = np.ones(L)
predvar = pow(10, -ROUNDOFF) * np.sqrt(N / 12.)
nth_predicted_var *= predvar
nth_positive_ci = 1.96 * nth_predicted_var
nth_negative_ci = -nth_positive_ci

for nth_simulation in range(L):
    round_errors = np.zeros(N)
    # Choose a random starting value
    y_0 = np.random.uniform(0.4, 0.6)

    # Computing the solution
    #x_true = np.zeros(N + 1)
    y_full = np.zeros(N + 1)
    y_full[0] = y_0
    #x_true[0] = y_0
    for n in range(N):
        y_full[n+1] = y_full[n] + h * a(y_full[n], n*h)
#    x_true[n+1] = x_sol(h * (n+1))
        round_errors[n] = y_full[n+1] - round(y_full[n+1], ROUNDOFF)

    print(f"{y_full}")

    nth_error[nth_simulation] = np.sum(round_errors)
    nth_var[nth_simulation] = np.std(round_errors)
    nth_means[nth_simulation] = np.mean(round_errors)


plt.plot(nth_attempt, nth_error, label = 'CRE')
plt.plot(nth_attempt, nth_var, label = 'CRE variance')
plt.plot(nth_attempt, nth_means, label = 'CRE mean')
plt.plot(nth_attempt, nth_predicted_var, label = 'CRE predicted var')
plt.plot(nth_attempt, nth_positive_ci, label = 'CRE predicted ci-')
plt.plot(nth_attempt, nth_negative_ci, label = 'CRE predicted ci-')
plt.title("Comulative Roundoff Errors distr,repeated simulations, N="+str(N))
plt.xlabel("nth-simulation")
plt.legend()
plt.grid()
plt.show()


#print(f"Analysis of the roundoff errors")
#print(f"They are: {round_errors}")

#print(f"Expected mean: 0.")
#print(f"Expected variance: {np.sqrt(pow(10, -2*ROUNDOFF)/12.)}")

#print(f"Actual mean: {np.mean(round_errors)}")
#print(f"Actual variance: {np.std(round_errors)}")


#print(f"Analyzing the CUMULATIVE roundoff errors")
#print(f"Actual value: {np.sum(round_errors)}")
#ci = 1.96 * pow(10, -ROUNDOFF) * np.sqrt(N / 12.)
#print(f"Predicted confidence interval: +-{ci}")
#print(f"Is in the ci? ")
#if (np.sum(round_errors) > -ci and np.sum(round_errors) < ci):
#    print("Yes!")
#else:
#    print("No...")
