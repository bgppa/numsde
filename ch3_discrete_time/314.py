# Numerically investigating the global truncation error for the Heun scheme
# We basically use Euler to predict a value, then trapezoidal rule
# to balance it
# x_true is the true analytical solution
# x_sol its pointwise evaluation
# y_sol its numerical approximation

import numpy as np
import matplotlib.pyplot as plt

# Artificial roundoff so to simulate numerical instability
ROUNDOFF = 5

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

max_cf = 12
cf_range = list(range(max_cf))
global_errors = []

for cf in cf_range:
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
        y_hat = y_sol[n] + h * a(y_sol[n], n * h)
        y_sol[n+1] = y_sol[n] + (a(y_sol[n], n*h) + a(y_hat, (n+1)*h)) * h/2.

        # ARTIFICIALLY ROUND UP THE SOLUTION SO TO STUDY NUM ERRORS
        y_sol[n+1] = round(y_sol[n+1], ROUNDOFF)
        x_sol[n+1] = x_true((n+1) * h)

    e_glb = np.abs(x_sol[N] - y_sol[N])
    print(f"Global Error: {e_glb : .3e}")
    global_errors.append(np.log(e_glb))

# Plot the results
#plt.plot(x_mesh, x_sol, label = 'true')
#plt.plot(x_mesh, y_sol, label = 'num')
#plt.title("Euler method, problem 1")
plt.plot(cf_range, global_errors, label = 'global err')
plt.plot(cf_range, [-x for x in cf_range], label = '-1 slope')
plt.plot(cf_range, [-2. * x for x in cf_range], label = '-2 slope')
plt.plot(cf_range, [-3. * x for x in cf_range], label = '-3 slope')
plt.title("Studying Heun Method, of order 2")
plt.legend()
plt.grid()
plt.show()


