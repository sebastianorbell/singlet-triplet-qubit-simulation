"""

Created by sebastian.orbell at 19.10.2022

"""
import numpy as np
import matplotlib.pyplot as plt
times = [5, 10, 15, 20]
#
gaussian = np.array([[0.968542, 0.02579471],
                     [0.9878413, 0.00949208],
                     [0.9758675, 0.01452975],
                     [0.98158824, 0.0131699]])

extended_gaussian = np.array([[0.99928916, 0.00032974],
                            [0.9987272, 0.00023681],
                            [0.9992568, 0.00038639],
                            [0.9949398, 0.00262975]])

cosine = np.array([[0.945945, 0.02585581],
                    [0.984101, 0.01357475],
                    [0.9716994, 0.0095069],
                    [0.98094165, 0.0160696]])

extended_cosine = np.array([[0.99899423, 0.00051613],
                            [0.99259126, 0.00665487],
                            [0.9860302, 0.01137743],
                            [0.9993207, 0.00057331]])

# gaussian = np.array([[0.9887650, 0.004534568],
#                      [0.998345, 0.0009770],
#                      [0.995009660, 0.00451563252],
#                      [0.9860732555, 0.0069967014]])
#
#
# extended_gaussian = np.array([[0.995787382, 0.0030491],
#                             [0.99975764751, 0.000123],
#                             [0.99227, 0.0044936],
#                             [0.9942100048, 0.0031919511]])
#
# cosine = np.array([[0.991659, 0.003633171],
#                     [0.9994544, 0.00029550],
#                     [0.99950921, 0.0000351663],
#                     [0.97840183, 0.0203679]])
#
# extended_cosine = np.array([[0.9977816343, 0.00150091946],
#                             [0.999706447, 0.000152768057],
#                             [0.9976819753, 0.00025644898],
#                             [0.9763648509, 0.007948695]])
#
# pwc5 = np.array([[0.996749 , 0.002616438],
#                             [0.9992410 , 0.00033617],
#                             [0.993256032 , 0.0028583295],
#                             [0.99443721 , 0.0013422266]])
#
# pwc50 = np.array([[0.99965286 , 0.000274447433],
#                             [0.9985660 , 0.0001597],
#                             [0.998691260 , 0.0010167937],
#                             [0.99804019 , 0.00085453409]])

# cosine = np.log(cosine)
# gaussian = np.log(gaussian)
# extended_gaussian = np.log(extended_gaussian)
# extended_cosine = np.log(extended_cosine)

fig, [ax1, ax2] = plt.subplots(nrows=2)
ax1.plot(times, gaussian[:, 0], label='Gaussian')
ax1.plot(times, extended_gaussian[:, 0], label='Extended Gaussian')
ax1.plot(times, cosine[:, 0], label='Cosine')
ax1.plot(times, extended_cosine[:, 0], label='Extended Cosine')
# ax1.plot(times, pwc5[:, 0], label='PWC 5')
# ax1.plot(times, pwc50[:, 0], label='PWC 50')
ax1.set_ylabel('Fidelity')
ax1.set_xlabel('Pulse time (ns)')
ax1.set_ylim(0.97, 1.0001)
ax2.plot(times, gaussian[:, 1], label='Gaussian')
ax2.plot(times, extended_gaussian[:, 1], label='Extended Gaussian')
ax2.plot(times, cosine[:, 1], label='Cosine')
ax2.plot(times, extended_cosine[:, 1], label='Extended Cosine')
# ax2.plot(times, pwc5[:, 1], label='PWC 5')
# ax2.plot(times, pwc50[:, 1], label='PWC 50')
ax2.set_ylabel('Leakage')
ax2.set_xlabel('Pulse time (ns)')
plt.legend()
plt.tight_layout()
plt.show()

