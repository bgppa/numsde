'''
An exercise to study the distribution of the CUMULATED roundoff error.
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
L = 2
powerN = 20


# Choose a starting value
y_0 = 1.
N = pow(2, powerN)
h = 1. / N

round_errors = np.zeros(N)


# Computing the solution
x_true = np.zeros(N + 1)
y_full = np.zeros(N + 1)
y_full[0] = y_0
x_true[0] = y_0
for n in range(N):
    y_full[n+1] = y_full[n] + h * a(y_full[n], n*h)
    x_true[n+1] = x_sol(h * (n+1))
    round_errors[n] = y_full[n+1] - round(y_full[n+1], ROUNDOFF)

tmesh = np.linspace(0, 1, N + 1)

print(f"{y_full}")

plt.plot(tmesh, x_true, label = 'true')
plt.plot(tmesh, y_full, label = 'num')
plt.legend()
plt.grid()
#plt.show()


print(f"Analysis of the roundoff errors")
print(f"They are: {round_errors}")

print(f"Expected mean: 0.")
print(f"Expected variance: {np.sqrt(pow(10, -2*ROUNDOFF)/12.)}")

print(f"Actual mean: {np.mean(round_errors)}")
print(f"Actual variance: {np.std(round_errors)}")


print(f"Analyzing the CUMULATIVE roundoff errors")
print(f"Actual value: {np.sum(round_errors)}")
ci = 1.96 * pow(10, -ROUNDOFF) * np.sqrt(N / 12.)
print(f"Predicted confidence interval: +-{ci}")
print(f"Is in the ci? ")
if (np.sum(round_errors) > -ci and np.sum(round_errors) < ci):
    print("Yes!")
else:
    print("No...")
