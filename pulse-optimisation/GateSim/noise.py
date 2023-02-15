"""

Created by sebastian.orbell at 16.9.2022

"""

import jax.numpy as np
from jax import random
import matplotlib.pyplot as plt

# key = random.PRNGKey(758493)  # Random seed is explicit in JAX

def noise_psd(N, key, psd=lambda f: 1):
    X_white = np.fft.rfft(random.normal(key, [N]))
    S = psd(np.fft.rfftfreq(N))
    # Normalize S
    S = S / np.sqrt(np.mean(S ** 2))
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)


def PSDGenerator(f):
    return lambda N, key: noise_psd(N, key, f)


@PSDGenerator
def white_noise(f):
    return 1


@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)


@PSDGenerator
def violet_noise(f):
    return f


@PSDGenerator
def pink_noise(f, alpha=1):
    return 1 / np.where(f == 0, float('inf'), np.power(f, alpha))

if __name__=='__main__':
    n = white_noise(int(500000))
    plt.plot(n)
    plt.show()