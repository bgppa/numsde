'''
Comparing a Geometric Brownian Motion solution to the actual one
'''
import numpy as np
import matplotlib.pyplot as plt


cfa = 1.
cfb = 1.3

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
    max2 = 6
    powers = np.array([2**x for x in range(min2, max2)])
    # for each fixed mesh, repeat simulations to compute averages of interest
    numsimu = 10000

    # Store both the strong pathwise errors as well as the weak errors
    strong_errors = np.zeros(len(powers))
    conf_strong_errors = np.zeros(len(powers))

    weak_errors = np.zeros(len(powers))
    conf_weak_errors = np.zeros(len(powers))

    # Weak errors relies on exact expectation, strong on the full path
    true_expectation = np.exp(cfa)
  
    for i in range(len(powers)):
        print(f"SIMULATIONS FOR {powers[i]} DISCR")
        ntimediv = powers[i]
        # Number of final discretization points; if divide in 2, obtain 3
        N = ntimediv + 1
        # time step
        h = 1. / ntimediv
        # Containers for the numerical approximations and true sol
        y_appx = np.zeros(N)
        y_true = np.zeros(N)
        x_axis = np.zeros(N)
        # Starting value
        y_appx[0] = 1.
        y_true[0] = y_appx[0]
        
        loc_strong_err = np.zeros(numsimu)
        loc_weak_err = np.zeros(numsimu)
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


            # Qui potrebbe essere il momento giusto per inserire un grafico
            if i == 8:
                plt.plot(x_axis, y_true, label="true")
                plt.plot(x_axis, y_appx, label="approx")
                plt.title(str(f"Mesh: {h}, simulation {k+1}/{numsimu}"))
                plt.grid()
                plt.legend()
                plt.show()

            loc_strong_err[k] = np.abs(y_true[-1] - y_appx[-1])
            loc_weak_err[k] = y_appx[ntimediv]


        strong_errors[i] = np.mean(loc_strong_err)
        conf_strong_errors[i] = 1.96 * np.std(loc_strong_err) / np.sqrt(numsimu)
        weak_errors[i] = np.mean(loc_weak_err) - true_expectation
        conf_weak_errors[i] = 1.96 * np.std(loc_weak_err) / np.sqrt(numsimu)


        print(f"strong_error = {strong_errors[i]}")
        print(f"ci: {conf_strong_errors[i]}")
        print(f"pure weak error: {weak_errors[i]}")
        print(f"ci: {conf_weak_errors[i]}")


    # Plot STRONG errors and their confidence intervals wrt meshsize
    plt.plot(np.log2(powers), np.log2(strong_errors), 
        label="strong error (CI dotted)", color='red')
    plt.plot(np.log2(powers), np.log2(strong_errors + conf_strong_errors), 
        color='red', linestyle='dotted')
    plt.plot(np.log2(powers), np.log2(strong_errors - conf_strong_errors), 
        color='red', linestyle='dotted')
    # Plot WEAK errors
    plt.plot(np.log2(powers), np.log2(np.abs(weak_errors)),
            label='weak error (CI not available - folded Normal)', 
            color='green')
    # Plot lines for references
    plt.plot(np.log2(powers), -np.log2(powers), label="-1 (compare to weak)",
            linestyle='dashed')
    plt.plot(np.log2(powers), -0.5 * np.log2(powers), 
        label="-1/2 (compare to strong)", linestyle='dashed')
    plt.xlabel("log2(Mesh cardinality)") 
    plt.ylabel("log2(Average global error)")
    plt.title("Error Behaviour of Euler SDE on GBM")
    plt.legend()
    plt.grid()
    plt.show()
