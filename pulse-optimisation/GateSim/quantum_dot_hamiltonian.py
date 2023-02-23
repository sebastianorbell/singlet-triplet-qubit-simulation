"""

Created by sebastian.orbell at 14.9.2022

"""

import jax.numpy as np
import qutip as qt
from typing import Callable, Optional

def return_initial_state(index: int, N: int) -> np.ndarray:
    """
    Return the state vector of size 'N'
    for a pure qubit state defined at 'index'.
    """
    
    initial_state = np.zeros(N).astype(complex)
    initial_state = initial_state.at[index].set(1.0+0.0j)
    return initial_state

class Driven_Two_Qubit:
        @classmethod
        def gen_drift_hamiltonian(cls,
            Bhz,
            Brz,
            Blz,
            J
        ) -> np.ndarray:
            """
            Define the drift Hamiltonian for a two qubit quantum dot system.
            """
            Ez = Bhz + (Brz+Blz)/2
            dEz = Brz - Blz
            return np.array(
                [
                    [Ez, 0, 0, 0],
                    [0, -((dEz/2) + (J/2)), J/2, 0],
                    [0, J/2, (dEz/2)-(J/2), 0],
                    [0, 0, 0,  -Ez]
                ]
            )

        @classmethod
        def gen_control_hamiltonion(cls,
                I: Callable,
                Q: Callable,
                Bry_0,
                Bly_0,
                Bry_1,
                Bly_1,
                omega: float,
                theta: float,
                t: np.ndarray,
        ) -> np.ndarray:
            """
            Define the control Hamiltonian for a driven superconducting single qubit system.

            """
            Bry = Bry_0 + Bry_1 * (I*np.sin(omega*t + theta)+Q*np.cos(omega*t + theta))
            Bly = Bly_0 + Bly_1 * (I*np.sin(omega*t + theta)+Q*np.cos(omega*t + theta))

            return np.array(
                [
                    [0, -1j*Bry, -1j*Bly, 0],
                    [1j*Bry, 0, 0, -1j*Bly],
                    [1j*Bly, 0, 0, -1j*Bry],
                    [0, 1j*Bly, 1j*Bry, 0]
                ]
            )

def pink_spectrum(frequencies, frequency_cutoff):
    return 1 / (frequencies + frequency_cutoff)

