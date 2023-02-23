"""
Created on 16/02/2023
@author sebastian
"""
import matplotlib.pyplot as plt
import numpy as np

from quantum_dot_hamiltonian import *
from evolve import *

initial_states = np.array([return_initial_state(0, 4)])

times = np.arange(0, 100, 50e-3) * 1e-9

Bhz = 14e9 * 2  * np.pi
Brz = 4.410e9 * 2 * np.pi
Blz = 4.287e6 * 2 * np.pi
Bry_0 = 55e6 * 2 * np.pi
Bly_0 = 5e6 * 2 * np.pi
Bry_1 = 1.
Bly_1 = 1.

J = 19.7e6 * 2 * np.pi


I = lambda time: 0
Q = lambda time: 0

h0 = Driven_Two_Qubit.gen_drift_hamiltonian(
            Bhz,
            Brz,
            Blz,
            J
)

omega = 0.0
theta = 0.0

hamiltonian_callable = lambda time: h0 + Driven_Two_Qubit.gen_control_hamiltonion(
                                                                                I,
                                                                                Q,
                                                                                Bry_0,
                                                                                Bly_0,
                                                                                Bry_1,
                                                                                Bly_1,
                                                                                omega,
                                                                                theta,
                                                                                time
                                                                            )

# ys = Evolve.vectorised_matrix_evolve(times, initial_states, hamiltonian_callable)

# J = 20e9 * 2 * np.pi
h0j = lambda J: Driven_Two_Qubit.gen_drift_hamiltonian(
            Bhz,
            Brz,
            Blz,
            J
)

J = np.linspace(-30, 30, 500) * 2 * np.pi * 1e9
# Bhz = np.linspace(-30, 30, 500) * 2 * np.pi * 1e9
time = 0.0
energies = np.array([np.linalg.eigvalsh(h0j(exchange)) for exchange in J])

labels = ['uu', 'ud', 'du', 'dd']

for e, label in zip(energies.T, labels):
    plt.plot(J*(1e-9), e*(1e-9), label=label)

plt.legend()
plt.ylabel('Energy (GHz)')
plt.xlabel('J (GHz)')
plt.show()