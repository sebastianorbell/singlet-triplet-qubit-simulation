"""
Created on 08/06/2022
@author sebastian
"""
'''
Plotting functions

'''

import matplotlib.pyplot as plt
from .main import h
import jax.numpy as np


eps = np.linspace(-0.5, 10, 100)*1e-3
plank = 4.1357e-15
tc = 3.64e9 * plank
tso = 100e-9
omega = lambda x: -np.arctan((2*np.sqrt(2)*tc)/x)
delta_g = 2.04
sigma_g = 11.0
mu_b = 5.788e-5
b_field = 5e-3

t_ev = 6.582e-16

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

# eps = np.linspace(-0.01, 2.5, 100)*1e-3
# energy = [np.linalg.eigvalsh(hm) for hm in hamiltonian(eps)]
# energy = np.array(energy)
# time = np.linspace(0, 250e-9, 1000)
# f_minus = (energy[:, 2] - energy[:, 1])*2.4e14
# f_zero = (energy[:, 1] - energy[:, 0])*2.4e14
# extent = [time[0]*1e9, time[-1]*1e9, eps[0]*1e3, eps[-1]*1e3]
# x_minus = np.sin(f_minus[..., np.newaxis]*time[np.newaxis, ...])
# x_zero = np.sin(f_zero[..., np.newaxis]*time[np.newaxis, ...])
# ratio = 0.5
# fig, [ax0, ax1, ax2, ax3] = plt.subplots(figsize=[2*single_column, single_column], ncols=4, sharey=True)
# ax0.plot(f_minus*1e-6, eps*1e3, label=r'$S_G-T_-$')
# ax0.plot(f_zero*1e-6, eps*1e3, label=r'$S_G-T_0$')
# ax0.set_xlim(0, 500)
# ax1.imshow(0.5*(x_zero*(1-ratio)+x_minus*ratio), extent=extent, cmap='inferno',aspect='auto', origin='lower')
# ax2.imshow(x_minus, extent=extent, cmap='inferno',aspect='auto', origin='lower')
# ax3.imshow(x_zero, extent=extent, cmap='inferno',aspect='auto', origin='lower')
# ax0.set_ylabel('Detuning (meV)')
# ax0.set_xlabel('Frequency (MHz)')
# ax1.set_xlabel('Time (ns)')
# ax2.set_xlabel('Time (ns)')
# ax3.set_xlabel('Time (ns)')
# ax1.title.set_text(r'$\left((S_G-T_0)+(S_G-T_-)\right)/2$')
# ax2.title.set_text(r'$S_G-T_-$')
# ax3.title.set_text(r'$S_G-T_0$')
# ax0.legend(loc='upper right')
# plt.savefig(os.getcwd()+'/figs/oscillating_3.pdf')
# plt.tight_layout()
# plt.show()
# #
fig = plt.figure(figsize=[2*single_column, single_column])
energy = np.array(energy)
plt.plot(eps*1e3, (energy[:, 2] - energy[:, 1])*2.4e14*1e-6, label=r'$S_G-T_0$')
plt.plot(eps*1e3, -(energy[:, 0] - energy[:, 1])*2.4e14*1e-6, label=r'$S_G-T_-$')
plt.ylim(0, 500)
plt.ylabel('Frequency (MHz)')
plt.xlabel('Detuning (meV)')
plt.legend()
plt.tight_layout()
plt.savefig(os.getcwd()+'/figs/frequency_detuning_1.pdf')
plt.show()
#
#
# fig = plt.figure(figsize=[single_column*2, single_column])
# plt.plot(np.array(eps)*1e3, np.array(energy)*1e6, label=[r'$T_-$', r'$S_G$', r'$T_0$', r'$T_+$', r'$S_E$'])
# plt.ylabel(r'Energy ($\mu$ eV)')
# plt.xlabel(r'Detuning (meV)')
# plt.legend()
# plt.ylim(-5, 15)
# plt.tight_layout()
# # plt.savefig(os.getcwd()+'/figs/energy_spectrum_detuning.pdf')
# plt.show()
# #
eps = 0.002
#
# fig = plt.figure(figsize=[2*single_column, single_column])
b_field = np.linspace(-5, 5, 100)*1e-3
hamiltonian = [h(eps, tc, tso, omega(eps), delta_g, sigma_g, mu_b, b) for b in b_field]
energy = [np.linalg.eigvalsh(hm) for hm in hamiltonian]
energy = np.array(energy)
# plt.plot(b_field*1e3, energy*1e6, label=[r'$T_-$', r'$S_G$', r'$T_0$', r'$T_+$', r'$S_E$'])
# plt.xlabel('B field (mT)')
# plt.ylabel(r'Energy ($\mu$eV)')
# plt.ylim(-2, 2)
# plt.legend()
# plt.tight_layout()
# # plt.savefig(os.getcwd()+'/figs/energy_spectrum_field.pdf')
# plt.show()
#
fig = plt.figure(figsize=[2*single_column, single_column])
plt.plot(b_field*1e3, (energy[:, 2] - energy[:, 1])*2.4e14*1e-6, label=r'$S_G-T_0$')
plt.plot(b_field*1e3, -(energy[:, 0] - energy[:, 1])*2.4e14*1e-6, label=r'$S_G-T_-$')
plt.ylabel('Frequency (MHz)')
plt.xlabel('B field (mT)')
plt.legend()
plt.tight_layout()
plt.savefig(os.getcwd()+'/figs/frequency_field_1.pdf')
plt.show()

''' 
2d detuning and field frequency modulation
'''
# b_field = np.linspace(-5, 5, 100)*1e-3
# eps = np.linspace(0, 2.5, 100)*1e-3
# hamiltonian = [[h(x, tc, tso, omega(x), delta_g, sigma_g, mu_b, b) for b in b_field] for x in eps]
# energy = [[np.linalg.eigvalsh(hm) for hm in hamiltonian1] for hamiltonian1 in hamiltonian]
# energy = 1e-9*np.array(energy)/plank
#
# extent=[-5, 5, 0, 2.5]
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=[2*single_column, single_column])
# im1 = ax1.imshow(np.log(energy[..., 2] - energy[..., 1]), extent=extent, origin='lower', cmap='inferno', aspect='auto')
# im2 = ax2.imshow(np.log(-energy[..., 0] + energy[..., 1]), extent=extent, origin='lower', cmap='inferno', aspect='auto')
# divider1 = make_axes_locatable(ax1)
# cax1 = divider1.append_axes('top', size='5%', pad=0.4)
# plt.colorbar(im1, cax=cax1, label=r'$S_G-T_0$ Frequency log((GHz))', orientation='horizontal')
# divider2 = make_axes_locatable(ax2)
# cax2 = divider2.append_axes('top', size='5%', pad=0.4)
# plt.colorbar(im2, cax=cax2, label=r'$S_G-T_-$ Frequency (log(GHz))', orientation='horizontal')
# ax1.set_xlabel('B field (mT)')
# ax2.set_xlabel('B field (mT)')
# ax1.set_ylabel(r'detuning ($\mu$eV)')
# plt.tight_layout()
# plt.savefig(os.getcwd()+'/figs/spin-funnel-sim-hamiltonian.pdf')
# plt.show()

'''
tunnel coupling 
'''

eps = np.linspace(-0.5, 2.5, 100)*1e-3
tc = np.linspace(0.1, 5, 100)*1e-5
b_field = 5e-3
omega = lambda x, y: -np.arctan((2*np.sqrt(2)*y)/x)

# energy = [[np.linalg.eigvalsh(h(x, y, tso, omega(x, y), delta_g, sigma_g, mu_b, b_field)) for x in eps] for y in tc]
# energy = 1e-9*np.array(energy)/plank
# #
# extent=[eps[0]*1e3, eps[-1]*1e3, tc[0]*1e3, tc[-1]*1e3]
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=[2*single_column, single_column])
# im1 = ax1.imshow(np.log(energy[..., 2] - energy[..., 1]), extent=extent, origin='lower', cmap='inferno', aspect='auto')
# im2 = ax2.imshow(np.log(-energy[..., 0] + energy[..., 1]), extent=extent, origin='lower', cmap='inferno', aspect='auto')
# divider1 = make_axes_locatable(ax1)
# cax1 = divider1.append_axes('top', size='5%', pad=0.4)
# plt.colorbar(im1, cax=cax1, label=r'$S_G-T_0$ Frequency log((GHz))', orientation='horizontal')
# divider2 = make_axes_locatable(ax2)
# cax2 = divider2.append_axes('top', size='5%', pad=0.4)
# plt.colorbar(im2, cax=cax2, label=r'$S_G-T_-$ Frequency (log(GHz))', orientation='horizontal')
# ax1.set_xlabel('Detuning (meV)')
# ax2.set_xlabel('Detuning (meV)')
# ax1.set_ylabel(r'Tunnel coupling (meV)')
# plt.tight_layout()
# plt.savefig(os.getcwd()+'/figs/tc-detuning.pdf')
# plt.show()

# eps = 0.002
# tc = np.linspace(0.1, 5, 100)*1e-5
# b_field = np.linspace(-5, 5, 100)*1e-3
# omega = lambda x, y: -np.arctan((2*np.sqrt(2)*y)/x)
#
# energy = [[np.linalg.eigvalsh(h(eps, y, tso, omega(eps, y), delta_g, sigma_g, mu_b, b)) for b in b_field] for y in tc]
# energy = 1e-9*np.array(energy)/plank
# #
# extent=[b_field[0]*1e3, b_field[-1]*1e3, tc[0]*1e3, tc[-1]*1e3]
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=[2*single_column, single_column])
# im1 = ax1.imshow(np.log(energy[..., 2] - energy[..., 1]), extent=extent, origin='lower', cmap='inferno', aspect='auto')
# im2 = ax2.imshow(np.log(-energy[..., 0] + energy[..., 1]), extent=extent, origin='lower', cmap='inferno', aspect='auto')
# divider1 = make_axes_locatable(ax1)
# cax1 = divider1.append_axes('top', size='5%', pad=0.4)
# plt.colorbar(im1, cax=cax1, label=r'$S_G-T_0$ Frequency log((GHz))', orientation='horizontal')
# divider2 = make_axes_locatable(ax2)
# cax2 = divider2.append_axes('top', size='5%', pad=0.4)
# plt.colorbar(im2, cax=cax2, label=r'$S_G-T_-$ Frequency (log(GHz))', orientation='horizontal')
# ax1.set_xlabel('B field (mT)')
# ax2.set_xlabel('B field (mT)')
# ax1.set_ylabel(r'Tunnel coupling (meV)')
# plt.tight_layout()
# plt.savefig(os.getcwd()+'/figs/tc-field.pdf')
# plt.show()