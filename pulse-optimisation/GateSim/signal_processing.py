"""

Created by sebastian.orbell at 12.9.2022

"""

import jax.numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def rise_and_fall(fx, x, xmax, width=1.e-1, window_gradient=1000., max_amplitude=20.):

    s_down = lambda t, g: 1/(1+np.exp(g*t))
    s_up = lambda t, g: 1 - s_down(t, g)
    s_amp = lambda signal, l, u: (2.*s_up((signal - 0.5*(u+l))/(0.5*(u-l)), 0.5)-1)*0.5*(u-l) + 0.5*(u+l)

    return s_up((x - width)/xmax, window_gradient)*s_down((x-(xmax-width))/xmax, window_gradient)*s_amp(fx, -max_amplitude, max_amplitude)
def discretise_pulse(f: Callable, times: np.ndarray, dt: float, central_frequency: float=0., bandwidth: float=3., sample_rate: float=2.) -> Callable:
    """
    Convolve with a sinc function, removing frequency components
    above the cuttoff and resampling to discretise the envelope.
    :param f:
    :param times:
    :return:
    """

    fx = f(times)
    # resample = 1/(dt*sample_rate)
    resample = 1
    # gx = rise_and_fall(fx, times, times[-1])
    gx = fx
    signal = resample_and_convolve(gx, times, fc=central_frequency, B=bandwidth, subsample=int(resample))
    g = lambda t: np.interp(t, times, signal[:times.__len__()])

    return g

def butterworth_transfer_function(low_cut_off, high_cut_off, frequency, n=3):
    return butterworth_low_pass(high_cut_off, frequency, n=n) * butterworth_high_pass(low_cut_off, frequency, n=n)

def butterworth_low_pass(cut_off, frequency, n=3):
    return 1. / np.sqrt(1 + np.power(frequency / cut_off, 2 * n))

def butterworth_high_pass(cut_off, frequency, n=1):
    return 1. - 1. / np.sqrt(1 + np.power(frequency / cut_off, 2 * n))

def fourier_domain(obs, obs_points, frequency_min = 0.2, frequency_max = 2, n=2000):
    frequencies = np.linspace(frequency_min, frequency_max, n)
    time_span = obs_points.max()-obs_points.min()
    ft = _dtft(obs, time_span, frequencies)
    return frequencies, ft

def _frequencies_to_periods(time_span, frequencies):
    return time_span * frequencies

def _dtft(Z, time_span, frequencies):
    periods = _frequencies_to_periods(time_span, frequencies)
    Z = Z.copy()
    mean = np.nanmean(Z)

    Z = Z.at[np.isnan(Z)].set(mean)

    n_0 = np.arange(0, Z.shape[0]) / Z.shape[0]
    z_0 = np.exp(-2j * np.pi * periods[:, np.newaxis] * n_0[np.newaxis, :])

    data = np.einsum('a, ba -> b', Z, z_0, optimize='greedy')
    return np.abs(data) / Z.size

def to_time_domain(frequency_signal, times):
    time_signal = np.real(np.fft.ifft(frequency_signal))
    return times, time_signal

def resample_and_convolve(fx, x, fc=2 * np.pi, B=np.pi, subsample=7):
    """
    To ensure that the pulse envelope which enters the Hamiltonian
    is realistic. The envelope, is frequency shifted so that the
    band center frequency is at 0Hz, convolved with a
    Convolve pulse with a sinc function,
    with a cutoff frequency fs. And subsample
    with a ratio 'subsample'.
    https://www.cl.cam.ac.uk/teaching/1213/DSP/slides-4up.pdf (slide 62)

    :param f:
    :param x:
    :param fc:
    :param subsample:
    :return:
    """

    delta = x[1].squeeze()-x[0].squeeze()
    gx = np.repeat(np.real(fx)[::subsample], subsample)
    y = gx*np.exp(-2j*np.pi*fc*x)
    N = 1000
    a = 0.5*N*delta
    B *= 2.*np.pi
    x1 = np.linspace(-a, a, N)
    sinc = B*np.sin(x1 * B) / (np.pi * x1 * B)
    s = np.convolve(y, sinc, 'same') / np.sum(sinc)
    s *= np.exp(2j*np.pi*fc*x)
    return s

def analyse_pulse_envelope(pulse_function, params, discretise_pulse):
    I, Q, times, dt = pulse_function(params)
    I_d = discretise_pulse(I, times, dt, sample_rate=5)
    Q_d = discretise_pulse(Q, times, dt, sample_rate=5)

    ts = np.linspace(0, times.max(), times.__len__() * 2)

    freq_I, response_I = fourier_domain(I(ts), ts)
    freq_I_d, response_I_d = fourier_domain(I_d(ts), ts)
    freq_Q, response_Q = fourier_domain(Q(ts), ts)
    freq_Q_d, response_Q_d = fourier_domain(Q_d(ts), ts)

    fig, axes = plt.subplots(nrows=2, ncols=2)

    axes[0][0].plot(ts, I(ts))
    axes[0][0].plot(ts, I_d(ts), '--')

    axes[0][1].plot(freq_I, (np.abs(response_I)))
    axes[0][1].plot(freq_I_d, (np.abs(response_I_d)), '--')

    axes[1][0].plot(ts, Q(ts))
    axes[1][0].plot(ts, Q_d(ts), '--')

    axes[1][1].plot(freq_Q, (np.abs(response_Q)))
    axes[1][1].plot(freq_Q_d, (np.abs(response_Q_d)), '--')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    Fs = 400                         # sampling rate
    Ts = 1.0/Fs                      # sampling interval
    ts = np.arange(0, 20, Ts)            # time vector

    f1 = 5
    f2 = 15

    test_signal = lambda ts: np.sin(2*np.pi*f1*ts) + np.sin(2*np.pi*f2*ts)
    frequencies, ft = fourier_domain(test_signal(ts), ts)

    signal_f = resample_and_convolve(test_signal, ts, fc=10, B=10, subsample=1)
    frequencies, ft_f = fourier_domain(signal_f, ts)

    fig, axes = plt.subplots(nrows=2)
    axes[0].plot(ts, test_signal(ts))
    axes[0].plot(ts, signal_f, '--')
    axes[0].set_xlabel('Time (ns)')
    axes[0].set_ylabel('Signal amplitude (a.u.)')

    axes[1].plot(frequencies, np.abs(ft), label='Signal')
    axes[1].plot(frequencies, np.abs(ft_f), '--', label='Bandlimited signal')
    axes[1].set_xlabel('Frequency (MHz)')
    axes[1].set_ylabel('Frequency response (a.u.)')
    axes[1].legend()

    plt.show()

    from pulse_parameterisation import gaussian
    I = gaussian(ts, 1., x_max=20)
    f, fft = fourier_domain(I, ts, frequency_min=-2, frequency_max=2)
    fig, axes = plt.subplots(nrows=2)
    axes[0].plot(ts, I)
    axes[1].plot(f, fft)
    plt.show()
