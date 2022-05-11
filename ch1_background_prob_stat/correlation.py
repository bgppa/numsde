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

    # Define the distribution Z = (x - mu_x) (y - mu_y)
    samples_z = (samples_x - mu_x) * (samples_y - mu_y)
    mu_z = np.mean(samples_z)

    # Return its mean divided by the previous standard deviations
    return mu_z / (std_x * std_y)
#---




if __name__ == '__main__':
    # Let's simulate two Gaussians having a desired correlation coefficient
    for rho in np.linspace(-1, 1, 10):
        n_samples = 500
        gaussian_1 = np.random.normal(0., 1., n_samples)
        tmp = np.random.normal(0., 1., n_samples)
        gaussian_2 = np.sqrt(1. - rho**2) * tmp + rho * gaussian_1

        print(f"\n(1): {np.mean(gaussian_1):.2f} {np.std(gaussian_1):.2f}")
        print(f"(2): {np.mean(gaussian_2):.2f} {np.std(gaussian_2):.2f}")
        print(f"Expected rho: {rho:.2f}")
        print(f"Estimated: {corr(gaussian_1, gaussian_2):.2f}")

        #Use also the numpy estimator as a double-check
        x_numpy = np.zeros((2, n_samples))
        x_numpy[0] = gaussian_1
        x_numpy[1] = gaussian_2
        print(f"With numpy: {np.corrcoef(x_numpy)[0][1]:.2f}")
