"""

Created by sebastian.orbell at 12.9.2022

"""
import jax.numpy as np

import matplotlib.pyplot as plt

def hermite(x,n):
    if n==0:
        return np.ones_like(x)
    elif n==1:
        return 2*x
    else:
        return 2*x*hermite(x,n-1)-2*(n-1)*hermite(x,n-2)

def gaussian(x, amp, *args, mu=10., sig=3.):
    return amp*(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

def recur_factorial(n):
   if n == 1:
       return n
   else:
       return n*recur_factorial(n-1)

def hermite_series(weights, x, mu):
    y = []
    x = x-mu
    for n, weight in enumerate(weights):
        value = weight*normalise_factor(n)*hermite(x, n)
        y.append(weight*normalise_factor(n)*hermite(x, n))*gaussian(x, 1, mu=10, sig=1))
        print(value.shape)
    return np.array(y)

normalise_factor = lambda n: 1 if n==0 else np.power(2., -n/2.)*(1./np.sqrt(recur_factorial(n)))

# x = np.linspace(-7, 7, 100)
#
# amp = 1
# ns = [0, 2, 4, 12]
# for n in ns:
#     nm = normalise_factor(n)
#     plt.plot(x, np.power(nm, 1)*hermite(x, n)*gaussian(x, amp), label=str(n))
# plt.legend()
# plt.show()

x = np.linspace(5,  15, 100)
weights = [1, 1, 1, 1]
y = hermite_series(weights, x, mu=0)
for index, item in enumerate(weights):
    plt.plot(x, y[index])
# plt.plot(x, np.sum(y, axis=0), '--')
plt.show()