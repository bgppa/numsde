'''
A simple correlation estimation to play a little bit with rv
'''
import numpy as np


def corr(samples_x, samples_y):
    '''
    Simple correlation estimator
    '''
    mu_x = np.mean(samples_x)
    std_x = np.std(samples_x)
    mu_y = np.mean(samples_y)
    std_y = np.std(samples_y)

    samples_z = (samples_x - mu_x) * (samples_y - mu_y)
    mu_z = np.mean(samples_z)

    print(f"Just cov: {mu_z:.2f}")
    return mu_z / (std_x * std_y)
#---


if __name__ == '__main__':
    h = 23
    N = 200000
    G1 = np.random.normal(0., 1., N)
    G2 = np.random.normal(0., 1., N)

    x1 = np.sqrt(h) * G1
    x2 = 0.5 * (np.sqrt(h) ** 3) * (G1 + (G2 / np.sqrt(3)))
    
    sig1 = np.std(x1)
    sig2 = np.std(x2)

#    sig_due = np.sqrt((h ** 3)/4. + 1. / 3.)
    sig_due = np.sqrt((h ** 3) / 3.)
    sig_uno = np.sqrt(h)

    print(f"Std of x2: {sig2:.3f} compare to {sig_due:.3f}")
    print(f"Std of x1: {sig1:.3f} compare to {sig_uno:.3f}")

#    print(x1)
#    print(x2)

    print(f"True covariance: {(h**2)/2:.2f}")
    print(f"Estimated correlation: {corr(x1, x2):.3f}")
    print(f"True correlation: {(h ** 2 / 2) / (sig1 * sig2): .3f}")
    print(f"Sig1: {sig1}")
    print(f"Sig2: {sig2}")
    
    z = np.zeros((2, N))
    z[0] = x1
    z[1] = x2
    print(f"Numpy estim.: {np.corrcoef(z)[0][1]: .2f}")
    
