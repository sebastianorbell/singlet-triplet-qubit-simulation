"""

Created by sebastian.orbell at 12.9.2022

"""
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt
import numpy as np
import matplotlib.pyplot as plt

t_max = 30
fs = 2.4
alpha = 0 * 0.6339
max_amp = 0.68
# ts = np.arange(-10, t_max + 10 + 1 / fs, 1 / fs)
ts = np.linspace(0, 30, 1000)

def gaussian(x, amp, *args, x_max = 20, mu=10., sig=3.):
    return amp*(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))-np.exp(-np.power(x_max - mu, 2.) / (2 * np.power(sig, 2.))))

def d_gaussian(x, amp, *args, mu=10., sig=3.):
    return -amp*(mu-x)/np.power(sig, 2.) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def scipy_filter(f, ts, fs=2.4):
    # The discretized pulse is modelled as a low-pass filtered step function
    pulse = f(ts)
    interp_Is = interp1d(ts, pulse, kind='nearest')

    sos_LP = butter(1, 20, btype='lp', fs=1e3 * fs, output='sos')  # specify low-pass filter
    fs_LP = sosfiltfilt(sos_LP, interp_Is(ts))  # apply low-pass filter

    fs_LP_interp = interp1d(ts, fs_LP, kind='cubic')

    g = lambda t: fs_LP_interp(t)

    return g

from GateSim.signal_processing import to_time_domain,to_frequency_domain, butterworth_transfer_function

def my_filter(f, ts):
    mean = np.mean(f(ts))
    print(mean)
    freq, response = to_frequency_domain(f(ts), ts)
    filtered = butterworth_transfer_function(20, 200, freq, n=1)*response
    times, filtered_signal = to_time_domain(filtered, ts)

    g = lambda t: np.interp(t, times, filtered_signal)

    return g

N = 3
omega = 2 * np.pi * 4.5
alpha = 1.5e0
phase = 0.
amp_I, amp_Q = 5.e-1, 5.e-1
max_t = 20.
mean, width = 10., 3.

s = lambda t: gaussian(t, amp_I, max_t, mean, width)
ds = lambda t: d_gaussian(t, amp_Q, mean, width)

w =  2 * np.pi * 4.5

I = lambda t: s(t) * (np.heaviside(t, 0) - np.heaviside(t - t_max, 1))
Q = lambda t: ds(t) * (np.heaviside(t, 0) - np.heaviside(t - t_max, 1))

fs = 1/(ts[1]-ts[0])

I_d = scipy_filter(I, ts, fs=fs)
Q_d = scipy_filter(Q, ts, fs=fs)

I_m = my_filter(I, ts)
Q_m = my_filter(Q, ts)


fig, [ax1, ax2] = plt.subplots(nrows=2)
# ax1.plot(ts, I(ts))
# ax1.plot(ts, I_d(ts))
ax1.plot(ts, I_m(ts), '--')
# ax2.plot(ts, Q(ts))
# ax2.plot(ts, Q_d(ts))
ax2.plot(ts, Q_m(ts), '--')
# ax1.legend()
# ax2.legend()
plt.show()