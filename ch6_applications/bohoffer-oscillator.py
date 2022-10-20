'''
Starting playing around with a simplified model for neuron firing.
'''
import numpy as np
import matplotlib.pyplot as plt


# ODE coefficients
a = 0.7
b = 0.8
c = 3.0
z = -0.34
start_val_x = -1.9
start_val_y = 1.2

# Add the noise to simulate statistical uncertainty
noise = 100

# First, before studying the trajectories themselves, we need to check
# we reach a stable regime
# we compute the solution for a very high mieshsize, then compare
ERR_REF = 12

powers = 2. ** np.array([ERR_REF,8, 9, 10, 11])

# Compute the numerical solution for this extreme case

strong_errors = np.zeros(len(powers) - 1)


y1_best_appx = np.zeros(2**ERR_REF + 1)
y2_best_appx = np.zeros(2**ERR_REF + 1)

for elm in range(len(powers)):
    print(f"Running simulation {elm+1}/{len(powers)}")
    N = int(powers[elm])
    T = 15.
    h = T/N
    print(f"{N} subdivisions, h={h}")
    y1_appx = np.zeros(N + 1)
    y1_appx[0] = start_val_x
    y2_appx = np.zeros(N + 1)
    y2_appx[0] = start_val_y

    for n in range(N):
        # Drift of the ODE
        tmp1 = c * (y1_appx[n] + y2_appx[n] - 1./3 * (y1_appx[n]**3) + z)
        tmp2 = (-1./ c) * (y1_appx[n] + b * y2_appx[n] - a)
        # Add the noise on the SDE case
        tmp1 += noise * np.sqrt(h) * np.random.normal()

        y1_appx[n+1] = y1_appx[n] + h * tmp1
        y2_appx[n+1] = y2_appx[n] + h * tmp2
        
        # Store the value at finest so to use them as benchmark for errors
        if elm == 0:
            ref_val = np.array(y1_appx[N], y2_appx[N])
            y1_best_appx = np.array(y1_appx)
            y2_best_appx = np.array(y2_appx)
        else:
            # Store the error computed by current compared to best
            current_final_val = np.array(y1_appx[N], y2_appx[N])
            strong_errors[elm - 1] = np.linalg.norm(current_final_val-ref_val)

#plt.plot(np.log2(powers[1:]), strong_errors, label="OBSERVED", color='red')
#plt.plot(np.log2(powers[1:]), -np.log2(powers[1:]), label="WISHED",
#        linestyle='dotted', color='blue')
#plt.grid()
#plt.title("Error behavior")
#plt.legend()
#plt.show()

# Let's plot the trajectories
plt.plot(np.linspace(0,T, 2**ERR_REF+1), y1_best_appx)
plt.plot(np.linspace(0,T, 2**ERR_REF+1), y2_best_appx)
plt.title(f"Neuron responses in current model [noise={noise}]")
plt.xlabel("t")
plt.ylabel("Voltage")
plt.grid()
plt.show()
           
plt.plot(y1_best_appx, y2_best_appx)
plt.title(f"Phase space [noise={noise}]")
plt.xlabel("Voltage1")
plt.ylabel("Voltage2")
plt.grid()
plt.show()




            
