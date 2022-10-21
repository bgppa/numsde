'''
Implementing an example of a Stochastic Volatility Model
'''
import torch as tc
import matplotlib.pyplot as plt

DOPLOT = True

N = 2 ** 7
T = 1.
h = tc.tensor(T / N)

q = 1.0
p = 0.3
alpha = 0.1

b_start = 1.
s_start = 1.
v_start = 0.1
c_start = 0.1


def r(t, b, s, v, c):
    '''
    Modeling the interest rate over time
    '''
    return 0.1
#---

b_appx = tc.zeros(N+1)
s_appx = tc.zeros(N+1)
v_appx = tc.zeros(N+1)
c_appx = tc.zeros(N+1)
x_axis = tc.zeros(N+1)

b_appx[0] = b_start
s_appx[0] = s_start
v_appx[0] = v_start
c_appx[0] = c_start

for n in range(N):

    r_val = r(n*h, b_appx[n], s_appx[n], v_appx[n], c_appx[n])

    b_appx[n+1] = b_appx[n] + h * r_val * b_appx[n]

    s_appx[n+1] = s_appx[n] + h * r_val * s_appx[n]
    tmp_s = (v_appx[n] * s_appx[n] * tc.sqrt(h)*tc.normal(0., 1., (1,)))
    s_appx[n+1] = s_appx[n+1] + tmp_s

    v_appx[n+1] = v_appx[n] + h * (-q) * (v_appx[n] - c_appx[n])
    tmp_v = p * v_appx[n] * tc.normal(0., 1., (1,))
    v_appx[n+1] = v_appx[n+1] + tmp_v

    c_appx[n+1] = c_appx[n] + h * (v_appx[n] - c_appx[n]) / alpha

    x_axis[n+1] = (n+1) * h # Just for debug to avoid wrong counting


if DOPLOT:
    plt.plot(x_axis, b_appx, label='bond')
    plt.plot(x_axis, s_appx, label='asset')
    plt.plot(x_axis, v_appx, label='volatility')
    plt.plot(x_axis, c_appx, label='mean reversion')
    plt.legend()
    plt.title("Simulating a Stochastic Volatility Model")
    plt.grid()
    plt.show()
