'''
Comparing a Geometric Brownian Motion solution to the actual one
'''
import numpy as np
import matplotlib.pyplot as plt


cfa = 1.5
cfb = 3.

def a(t, x):
    '''
    Deterministic drift
    '''
    return cfa * x


def b(t, x):
    '''
    Stochastic oscillation
    '''
    return cfb * x



if __name__ == '__main__':
    # Number of parts in which we subdivide the unitary time interval, e.g. 2


    min2 = 2
    max2 = 10
    powers_list = [2**x for x in range(min2, max2)]
    powers = np.array(powers_list)
    
    numsimu = 5000
    path_error = np.zeros(len(powers))
    conf_error = np.zeros(len(powers))
  
    for i in range(len(powers)):
        print(f"SIMULATIONS FOR {powers[i]} DISCR")
        ntimediv = powers[i]
        # Number of final discretization points; if divide in 2, obtain 3
        N = ntimediv + 1
        # time step
        h = 1. / ntimediv

        y_appx = np.zeros(N)
        y_true = np.zeros(N)
        x_axis = np.zeros(N)

        y_appx[0] = 1.
        y_true[0] = 1.
        
        loc_err = np.zeros(numsimu)
        for k in range(numsimu):
            w_collector = np.random.normal(0., 1., ntimediv) * np.sqrt(h)
            # Here comes the simulation
            for n in range(ntimediv):
                # compute the axis, it serves as a double check for indices
                x_axis[n+1] = h * (n+1)
                # numerical approximated solution
                y_appx[n+1] = y_appx[n] + a(n*h, y_appx[n]) * h
                y_appx[n+1] += b(n*h, y_appx[n]) * w_collector[n]

                # theoretical one
                # here t = (n+1) * h
                cf1 = (cfa - 0.5 * (cfb**2)) * (n+1) * h
                cf2 = cfb * np.sum(w_collector[:n+1])
                y_true[n+1] = y_true[0] * np.exp(cf1 + cf2)

            loc_err[k] = np.abs(y_true[-1] - y_appx[-1])

        path_error[i] = np.mean(loc_err)
        conf_error[i] = 1.96 * np.std(loc_err) / np.sqrt(numsimu)
        print(f"path_error = {path_error[i]}")
        print(f"ci: {conf_error[i]}")


    plt.plot(np.log(powers), np.log(path_error), 
        label="Pathwise Error with Euler", color='red')
    plt.plot(np.log(powers), np.log(path_error + conf_error), 
        color='red', linestyle='dotted')
    plt.plot(np.log(powers), np.log(path_error - conf_error), 
        color='red', linestyle='dotted')

    plt.plot(np.log(powers), -np.log(powers), label="-1 (ODE Euler)",
            linestyle='dotted')
    plt.plot(np.log(powers), -0.5 * np.log(powers), 
        label="-1/2 (SDE Euler)", linestyle='dotted')
    plt.xlabel("log(Mesh cardinality)") 
    plt.ylabel("log(Average global error)")
    plt.title("Estimating SDE Euler pathwise convergence")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the results
#    plt.plot(x_axis, y_appx, label="Euler SDE")
#   plt.plot(x_axis, y_true, label="True")
#   plt.grid()
#    plt.legend()
#    plt.show()
