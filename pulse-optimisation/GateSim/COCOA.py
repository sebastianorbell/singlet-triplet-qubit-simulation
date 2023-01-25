"""

Created by sebastian.orbell at 22.9.2022

"""

import jax.numpy as np
from jax import vmap
import matplotlib.pyplot as plt

from signal_processing import *

class COCOA:
    # TODO Make the frequency components harmonics of the timespan
    @classmethod
    def pwc(cls, params, times):
        repeats = times.__len__() / params.__len__()
        pwc_signal = np.repeat(params, int(repeats))
        return pwc_signal

    @classmethod
    def transform_to_fourier_series(cls, pwc_signal, times, fmax, Nc):
        frequencies, ft = fourier_domain(pwc_signal, times, frequency_min=0., frequency_max=fmax*2, n=2*Nc)
        ft = ft.at[Nc:].set(0.)
        return frequencies, ft

    @classmethod
    def transform_to_time_domain(cls, ft, frequencies, times):
        g = lambda x: ft[0] + np.sum(np.abs(ft)*np.cos(2. * np.pi * frequencies * x + np.angle(ft)))
        return vmap(g)(times)

if __name__=='__main__':
    times = np.linspace(0, 200, 1000)
    subsample = int(1000/50)
    params = np.repeat(np.sin(times)[::subsample], subsample)
    fmax = 100
    Nc = 500
    pwc_envelope = COCOA.pwc(params, times)
    frequencies, ft = COCOA.transform_to_fourier_series(pwc_envelope, times, fmax, Nc)
    time_domain = COCOA.transform_to_time_domain(ft, frequencies, times)

    plt.plot(times, pwc_envelope)
    plt.plot(times, time_domain, '--')
    plt.ylim(-1.4, 1.4)
    plt.show()

    plt.scatter(frequencies, ft)
    plt.show()
