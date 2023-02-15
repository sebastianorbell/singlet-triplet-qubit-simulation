import os

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from simulation.Qubit import Qubit
from simulation.functions import to_real, to_complex

font_size = 7
plt.rc('font', size=font_size) #controls default text size
plt.rc('font', size=font_size) #controls default text size
plt.rc('axes', titlesize=font_size) #fontsize of the title
plt.rc('axes', labelsize=font_size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=font_size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=font_size) #fontsize of the y tick labels
plt.rc('legend', fontsize=font_size) #fontsize of the legend

plt.rcParams['font.size'] = '7'
cm = 1 / 2.54
single_column = 8.6 * cm

eps = np.linspace(-1.0, 1, 100)*1e-3
plank = 4.1357e-15
tc = 3.64e9 * plank * 0.5
tso = 3000e-9
omega = lambda x: -np.arctan((2*np.sqrt(2)*tc)/x)
delta_g = 4.04
sigma_g = 11.0
mu_b = 5.788e-5
b_field = 6e-3

t_ev = 6.582e-16


def h(eps, tc, tso, omega, delta_g, sigma_g, mu_b, b_field):
    """
    Hamiltonian in the basis:
        (S_E, S_G, T_+, T_0, T_-)
    :param eps:
    :param tc:
    :param tso:
    :param omega:
    :param delta_g:
    :param sigma_g:
    :param mu_b:
    :param b_field:
    :return:
    """
    return np.array([
        [
            (eps/2) + np.sqrt(((eps/2)**2)+2*(tc**2)), 0, tso*np.cos(omega/2), -0.5*delta_g*mu_b*b_field*np.sin(omega/2), tso*np.cos(omega/2)
        ],
        [
            0, (eps/2) - np.sqrt(((eps/2)**2)+2*(tc**2)), tso*np.sin(omega/2), 0.5*delta_g*mu_b*b_field*np.cos(omega/2), tso*np.sin(omega/2)
        ],
        [
            tso*np.cos(omega/2), tso*np.sin(omega/2), 0.5*sigma_g*mu_b*b_field, 0, 0
        ],
        [
            -0.5*delta_g*mu_b*b_field*np.sin(omega/2), 0.5*delta_g*mu_b*b_field*np.cos(omega/2), 0, 0, 0
        ],
        [
            tso*np.cos(omega/2), tso*np.sin(omega/2), 0, 0, -0.5*sigma_g*mu_b*b_field
        ]
    ])

hamiltonian = lambda eps: np.squeeze(np.array([h(x, tc, tso, omega(x), delta_g, sigma_g, mu_b, b_field) for x in eps]))

energy = [np.linalg.eigvalsh(hm) for hm in hamiltonian(eps)]
#
eps_max = .2e-3
plt.plot(np.array(eps)*1e3, np.array(energy)*1e6, label=[r'$T_-$', r'$S_G$', r'$T_0$', r'$T_+$', r'$S_E$'])
plt.ylabel(r'Energy ($\mu$ eV)')
plt.xlabel(r'Detuning (meV)')
plt.axvline(eps_max, -10, 10)
plt.legend()
plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

initial_state = np.array(
    [
        0+0j,
        1.+0j,
        0+0j,
        0+0j,
        0+0j,
    ]
)

n_times = 4000
# eps_max = 1e-6
times = np.linspace(0, 190, n_times)*1e9*2
control = np.full_like(times, eps_max)
qubit = Qubit()
scale = 0.0000367 # convert to atomic units
hamiltonian = lambda x: scale*np.array(h(x, tc, tso, omega(x), delta_g, sigma_g, mu_b, b_field))

plt.imshow(hamiltonian(eps_max))
plt.colorbar()
plt.show()

ramp_time = 10e9
ramp_times = np.linspace(0, times[-1] + ramp_time, n_times + int(n_times*ramp_time/times[-1]))

eps = np.linspace(-0.1*eps_max, eps_max, 200)
ramp_up_eps = lambda x: np.linspace(-1e-6, x, int(n_times*ramp_time/times[-1]))
control = np.array([np.concatenate((ramp_up_eps(ep), np.full(n_times, ep)), axis=0) for ep in eps])


# control = np.full([n_times, 100], eps).T
ys = jax.vmap(qubit.map_matrix_evolve, in_axes=(None, 0, None, None))(ramp_times, control, to_real(initial_state), hamiltonian)

n1 = 70
n2 = -1
ramp_times = ramp_times*1e-9
fig, [ax1, ax2, ax3] = plt.subplots(figsize=[2*single_column, single_column], nrows=3)
ax1.plot(ramp_times, control[n1, :], 'k--', label=f'(a)')
ax1.plot(ramp_times, control[n2, :], 'r--', label=f'(b)')
ax1.legend()
ax2.plot(ramp_times, np.square(np.abs(np.squeeze(ys))).T[:, 0, n1], label=r'S_E')
ax2.plot(ramp_times, np.square(np.abs(np.squeeze(ys))).T[:, 1, n1], label=r'S_G')
ax2.plot(ramp_times, np.square(np.abs(np.squeeze(ys))).T[:, 2, n1], label=r'T_+')
ax2.plot(ramp_times, np.square(np.abs(np.squeeze(ys))).T[:, 3, n1], label=r'T_0')
ax2.plot(ramp_times, np.square(np.abs(np.squeeze(ys))).T[:, 4, n1], label=r'T_-')
ax3.plot(ramp_times, np.square(np.abs(np.squeeze(ys))).T[:, 0, n2], label=r'S_E')
ax3.plot(ramp_times, np.square(np.abs(np.squeeze(ys))).T[:, 1, n2], label=r'S_G')
ax3.plot(ramp_times, np.square(np.abs(np.squeeze(ys))).T[:, 2, n2], label=r'T_+')
ax3.plot(ramp_times, np.square(np.abs(np.squeeze(ys))).T[:, 3, n2], label=r'T_0')
ax3.plot(ramp_times, np.square(np.abs(np.squeeze(ys))).T[:, 4, n2], label=r'T_-')
ax3.set_xlabel('Time (ns)')
ax1.set_ylabel('Detuning (eV)')
ax2.set_ylabel('(a) State probability')
ax3.set_ylabel('(b) State probability')
plt.legend()
plt.savefig(os.getcwd()+'/figs/evolution_all_states_5.pdf')
plt.tight_layout()
plt.show()

extent = [ramp_times[0], ramp_times[-1], eps[0], eps[-1]]

energy = [np.linalg.eigvalsh(hamiltonian(x)) for x in eps]
energy = np.array(energy)

f_minus = (energy[:, 2] - energy[:, 1])*2.4e14
f_zero = (energy[:, 1] - energy[:, 0])*2.4e14

from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, [ax1, ax2, ax3] = plt.subplots(figsize=[2*single_column, single_column], ncols=3, sharey=True)
im1 = ax1.imshow(0.5*(np.square(np.abs(ys[:, 3, :]))+np.square(np.abs(ys[:, 4, :]))), cmap='inferno', extent=extent, aspect='auto', origin='lower')
im2 = ax2.imshow(np.square(np.abs(ys[:, 4, :])), cmap='inferno', extent=extent, aspect='auto', origin='lower')
im3 = ax3.imshow(np.square(np.abs(ys[:, 3, :])), cmap='inferno', extent=extent, aspect='auto', origin='lower')
ax1.set_ylabel('Detuning (meV)')
ax1.set_xlabel('Time (ns)')
ax2.set_xlabel('Time (ns)')
ax3.set_xlabel('Time (ns)')
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes('top', size='5%', pad=0.4)
plt.colorbar(im1, cax=cax1, label=r'$\left((S_G-T_0)+(S_G-T_-)\right)/2$', orientation='horizontal')
divider2 = make_axes_locatable(ax2)
cax1 = divider2.append_axes('top', size='5%', pad=0.4)
plt.colorbar(im1, cax=cax1, label=r'$S_G-T_-$', orientation='horizontal')
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes('top', size='5%', pad=0.4)
plt.colorbar(im3, cax=cax3, label=r'$S_G-T_0$', orientation='horizontal')
plt.savefig(os.getcwd()+'/figs/evolution_rabi_5.pdf')
plt.tight_layout()
plt.show()