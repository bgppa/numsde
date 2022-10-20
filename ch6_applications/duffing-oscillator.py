'''
Another experiment where we try to observe the diffeomorphism property.
'''
import numpy as np
import matplotlib.pyplot as plt

alpha = 1.
noise = 0.5

T = 100
N = 2 ** 14
h = T/N
epsilon = 0.2
kappas = np.array(range(10, 20, 3))


y1_appx = np.zeros(N + 1)
y2_appx = np.zeros(N + 1)

fig, (ax1, ax2) = plt.subplots(2)

for k in kappas:
    y1_appx = np.zeros(N + 1)
    y2_appx = np.zeros(N + 1)
    y1_appx[0] = -epsilon * k
    y2_appx[0] = 0.

    # Approximate everything with Euler
    for n in range(N):
        # y1 is the actual trajectory we want to study at the end
        y1_appx[n+1] = y1_appx[n] + h * y2_appx[n]

        # introduce the term y2 = y1' to solve the system more easily
        term2 = y1_appx[n] * (alpha - (y1_appx[n] ** 2)) - y2_appx[n]
        y2_appx[n+1] = y2_appx[n] + h * term2

        # Add the residual SDE term
        y2_appx[n+1] += noise * y1_appx[n] * np.sqrt(h) * np.random.normal()

    print(f"1st-component solution: {y1_appx}")
    print(f"2nd-component solution: {y2_appx}")

    ax1.plot(np.linspace(0, T, N+1), y1_appx)
    ax2.plot(y1_appx, y2_appx)


ax1.set_title(f"{N} subdivisions, mesh = {h}")
ax1.grid()
ax2.grid()
fig.suptitle("Noise can move trajectories from one attractor to another")
plt.show()
