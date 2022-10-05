# Using the Euler method on a simple ODE
# x_true is the true analytical solution
# x_sol its pointwise evaluation
# y_sol its numerical approximation

import numpy as np
import matplotlib.pyplot as plt

def a(x_n, t_n):
    '''
    Function driving the ODE
    '''
    return -5. * x_n
#---

def x_true(t):
    '''
    True solution
    '''
    return np.exp(-5. * t)
#---

cf = 5
N = 2 ** cf
h = 1./N
x0 = 1.

x_mesh = np.linspace(0, 1, N+1)

x_sol = np.zeros(N+1)
y_sol = np.zeros(N+1)
y_sol[0] = x0
x_sol[0] = x0

for n in range(N):
    # n goes from 0 to N-1, included
    y_sol[n + 1] = y_sol[n] + h * a(y_sol[n], n * h)
    x_sol[n + 1] = x_true((n+1) * h)

e_glb = np.abs(x_sol[N] - y_sol[N])
print(f"Global Error: {e_glb : .3e}")

# Plot the results
plt.plot(x_mesh, x_sol, label = 'true')
plt.plot(x_mesh, y_sol, label = 'num')
plt.title("Euler method, problem 1")
plt.legend()
plt.grid()
plt.show()


