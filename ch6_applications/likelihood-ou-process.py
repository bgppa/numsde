'''
Simulating a simple OU process and using the Signature as active role
to estimatie its driving parameter.
'''
import torch as tc
import signatory as sg
import matplotlib.pyplot as plt

alpha = tc.tensor(-1.)

N = 2 ** 6
T = tc.tensor(1.)
h = T / N

y_appx = tc.zeros(N + 1)
y_appx[0] = 0.

for n in range(N):
    y_appx[n+1] = y_appx[n] + h * alpha * y_appx[n]
    y_appx[n+1] = y_appx[n+1] + tc.sqrt(h) * tc.normal(0., 1., (1,))

plt.plot(tc.linspace(0, 1, N+1), y_appx, label=f"alpha={alpha}")
plt.title("OU Process")
plt.grid()
plt.legend()
plt.show()


# Now, extend the path and compute the signature
y_augmented = tc.zeros(1, N, 2)
y_augmented = 




