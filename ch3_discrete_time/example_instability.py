'''
A simple example showing the danger of instable numerical schemes.
Here we use the Euler criterium over the ode x' = -8x,
which lead to instability for time steps bigger than 2**-3 = 1/8
'''
import numpy as np
import matplotlib.pyplot as plt

subdivisions = np.array([5, 6, 7, 8, 10])

# Conduct a numerical approximation for every of the meshsizes above
for N in subdivisions:
    h = 1./N
    y_appx = np.zeros(N + 1)
    # Starting value
    y_appx[0] = 1.

    # Compute the numerical solution according to Euler scheme
    for nth in range(N):
        y_appx[nth+1] = (1. - 8.*h) * y_appx[nth]

    err = np.abs(y_appx[N] - np.exp(-8))
    mylabel = f"N={N}, err={err:.3e}"
    if N < 10:
        mylabel += " [INSTABLE]"
    else:
        mylabel += " [STABLE]"
    plt.plot(np.linspace(0., 1., N+1), y_appx, label=mylabel)
    
x_axis = np.linspace(0, 1, N)
y_true = np.exp(-8 * x_axis)
# When leaving the loop, N = 32
plt.plot(x_axis, y_true, label="true")
plt.grid()
plt.title("Example of instability when using Euler improperly")
plt.legend()
plt.show()




