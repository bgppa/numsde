'''
An exercise on analyzing the distribution of roundoff errors.
'''
import numpy as np

# Set the decimal rounding up to this decimal digit
ROUNDOFF = 5
NUMITER = 300

# 1. Run the selected iteration and collect all the rundoff errors
y = 0.1
errs = np.zeros(NUMITER)
for nth in range(NUMITER):
    # Precise value of the next iteration
    y_full = y * np.pi / 3.
    # ...but we round on purpose
    y = round(y_full, ROUNDOFF)
    # and store the error intended as simple difference
    errs[nth] = y_full - y

#
#print(f"List of errors: {errs}")

# I expect them to be uniformly distributed in the interval [-a, a]
# with a = 5 * 10 -(ROUNDOFF + 1)
# Therefore with mean = (a - a) / 2 = 0
# and standard deviation (a + a) / sqrt(12)
# I just compare these values instead of a deeper statistica analysis
print(f"Expected mean: {0.}")
print(f"Expected std: {pow(10, -ROUNDOFF)/ np.sqrt(12.)}")

print(f"Actual mean: {np.mean(errs)}")
print(f"Actual std: {np.std(errs)}")
