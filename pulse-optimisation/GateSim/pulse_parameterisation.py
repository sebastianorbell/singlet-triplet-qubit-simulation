"""

Created by sebastian.orbell at 14.9.2022

"""

import jax.numpy as np
from jax import random

# def gaussian(x, mu, sig, *args):
#     """Truncated Gaussian pulse envelope."""
#     return np.exp(-0.5*np.power((x - mu)/sig, 2))
#
# def d_gaussian(x, mu, sig, *args):
#     """Gaussian derivative pulse envelope."""
#     return -((x-mu)/sig)*gaussian(x, mu, sig, *args)

def gaussian(x, amp, *args, mu=10., sig=3.):
    """Truncated Gaussian pulse envelope."""
    return amp*(np.exp(-0.5*np.power((x - mu)/sig, 2)))

def d_gaussian(x, amp, *args, mu=10., sig=3.):
    """Gaussian derivative pulse envelope."""
    return -((x-mu)/sig)*gaussian(x,amp, mu=mu, sig=sig)

def d2_gaussian(x, amp, *args, mu=10., sig=3.):
    "Gaussian second derivative pulse envelope"
    return amp*((np.power(x - mu, 2.)/np.power(sig, 2))-1)*gaussian(x, mu=mu, sig=sig)

def d3_gaussian(x, amp, *args, mu=10., sig=3.):
    "Gaussian third derivative pulse envelope"
    return amp*((np.power(x - mu, 3.)/np.power(sig, 3))+3*((x-mu)/sig))*gaussian(x, mu=mu, sig=sig)

class Gaussian:
    @classmethod
    def random_initial(cls, scale, *, key=random.PRNGKey(0), max_t=16, **kwargs):
        amps = random.uniform(key, [1], minval=-scale, maxval=scale)
        pulse_params = amps
        return pulse_params

    @classmethod
    def construct(cls, pulse_params, *, max_t=16, N=24000):
        """Costruct a parameterised DRAG pulse envelope."""
        amp_I = pulse_params
        mean = max_t/2
        width = max_t/6
        times = np.linspace(0, max_t, N)
        dt = max_t/N
        I = lambda t: gaussian(t, amp_I, x_max=max_t, mu=mean, sig=width)
        Q = lambda t: d_gaussian(t, 0.0*amp_I, mu=mean, sig=width)
        return I, Q, times, dt


class DRAG:
    @classmethod
    def random_initial(cls, scale, *, key=random.PRNGKey(0), max_t=16, **kwargs):
        amps = random.uniform(key, [2], minval=-scale, maxval=scale)*np.array([1.0, 1.0e-1])
        # mean = max_t/2.
        # width = mean/3.
        pulse_params = amps
        return pulse_params

    @classmethod
    def construct(cls, pulse_params, *, max_t=16, N=6000):
        """Costruct a parameterised DRAG pulse envelope."""
        amp_I, amp_Q = pulse_params
        mean = max_t/2
        width = max_t/6
        times = np.linspace(0, max_t, N)
        dt = max_t/N
        I = lambda t: gaussian(t, amp_I, mu=mean, sig=width)
        Q = lambda t: d_gaussian(t, amp_Q, mu=mean, sig=width)
        return I, Q, times, dt

class Extended_DRAG:
    @classmethod
    def random_initial(cls, scale, *, key=random.PRNGKey(0), max_t=16, **kwargs):
        amps = random.uniform(key, [4], minval=-scale, maxval=scale)
        # mean = max_t/2.
        # width = mean/3.
        return amps

    @classmethod
    def construct(cls, pulse_params, *, max_t=16, N=6000):
        """Costruct a parameterised DRAG pulse envelope."""
        amp_0, amp_1, amp_2, amp_3 = pulse_params
        mean = max_t/2
        width = max_t/8
        times = np.linspace(0, max_t, N)
        dt = max_t/N
        I = lambda t: gaussian(t, amp_0, mu=mean, sig=width) + d2_gaussian(t, amp_2, mu=mean, sig=width)
        Q = lambda t: d_gaussian(t, amp_1, mu=mean, sig=width) + d3_gaussian(t, amp_3, mu=mean, sig=width)
        return I, Q, times, dt

class Cosine:
    @classmethod
    def random_initial(cls, scale, *, key=random.PRNGKey(0), max_t=16, **kwargs):
        pulse_params = random.uniform(key, [2], minval=-scale, maxval=scale)
        return pulse_params
    @classmethod
    def construct(cls, pulse_params, *, max_t=16, N=6000):
        times = np.linspace(0, max_t, N)
        dt = max_t / N
        params_I, params_Q = pulse_params

        I = lambda t: 0.5*params_I*(1.-np.cos((2.*np.pi*t)/max_t))
        Q = lambda t: 0.5*params_Q*(np.sin((2.*np.pi*t)/max_t))

        return I, Q, times, dt

class Slepian:
    @classmethod
    def random_initial(cls, scale, *, key=random.PRNGKey(0), max_t=16, **kwargs):
        pulse_params = random.uniform(key, [2], minval=-scale, maxval=scale)
        return pulse_params
    @classmethod
    def construct(cls, pulse_params, *, max_t=16, N=6000):
        times = np.linspace(0, max_t, N)
        dt = max_t / N
        params_I, params_Q = pulse_params

        I = lambda t: 0.5*params_I*(1.-np.cos((2.*np.pi*t)/max_t))
        Q = lambda t: 0.5*params_Q*(np.sin((2.*np.pi*t)/max_t))

        return I, Q, times, dt

class Extended_cosine:
    @classmethod
    def random_initial(cls, scale, *, key=random.PRNGKey(0), max_t=16, **kwargs):
        pulse_params = random.uniform(key, [4], minval=-scale, maxval=scale)
        return pulse_params
    @classmethod
    def construct(cls, pulse_params, *, max_t=16, N=6000):
        times = np.linspace(0, max_t, N)
        dt = max_t / N

        I = lambda t: 0.5*pulse_params[0]*(1.-np.cos(2.*np.pi*t/max_t))\
                      +0.5*pulse_params[1]*(1.-np.cos(2*2.*np.pi*t/max_t))

        Q = lambda t: 0.5*pulse_params[2]*(np.sin(2.*np.pi*t/max_t)) \
                      +0.5*pulse_params[3]*(np.sin(2*2.*np.pi*t/max_t))

        return I, Q, times, dt

class PWC:
    @classmethod
    def random_initial(cls, scale, *, key=random.PRNGKey(0), N_components=10, **kwargs):
        pulse_params = random.normal(key, [N_components*2])*scale
        return pulse_params

    @classmethod
    def construct(cls, pulse_params, *, max_t=16, N=7000):
        times = np.linspace(0, max_t, N)
        dt = max_t/N
        params_I, params_Q = np.array_split(pulse_params, 2)
        repeats = times.__len__() / params_I.__len__()
        pwc_I = np.repeat(params_I, int(repeats))
        pwc_Q = np.repeat(params_Q, int(repeats))

        I = lambda t: np.interp(t, times, pwc_I)
        Q = lambda t: np.interp(t, times, pwc_Q)

        return I, Q, times, dt


class Fourier:
    @classmethod
    def random_initial(cls, scale, *, key=random.PRNGKey(0), N_components=10, **kwargs):
        pulse_params = random.normal(key, [N_components*4+2])*(scale/np.sqrt(N_components))
        return pulse_params

    @classmethod
    def construct(cls, pulse_params, *, max_t=16, N=6000):
        times = np.linspace(0, max_t, N)
        dt = max_t / N
        params_I, params_Q = pulse_params.split(2)
        I_0, Q_0 = params_I[0], params_Q[0]
        I_sin, I_cos = params_I[1:].split(2)
        Q_sin, Q_cos = params_Q[1:].split(2)

        series_I = np.full_like(times, I_0)
        for n, (amplitude_cos, amplitude_sin) in enumerate(zip(I_cos, I_sin)):
            series_I += amplitude_cos * np.cos((n + 1) * (times * 2 * np.pi / max_t)) \
                        + amplitude_sin * np.sin((n + 1) * (times * 2 * np.pi / max_t))

        series_Q = np.full_like(times, Q_0)
        for n, (amplitude_cos, amplitude_sin) in enumerate(zip(Q_cos, Q_sin)):
            series_Q += amplitude_cos * np.cos((n + 1) * (times * 2 * np.pi / max_t)) \
                        + amplitude_sin * np.sin((n + 1) * (times * 2 * np.pi / max_t))

        I = lambda t: np.interp(t, times, series_I)
        Q = lambda t: np.interp(t, times, series_Q)

        return I, Q, times, dt