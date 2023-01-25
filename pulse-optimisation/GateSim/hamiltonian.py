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

class Driven_Single_Qubit:
        @classmethod
        def gen_drift_hamiltonian(cls,
            omega: float,
            alpha: float,
                N: int
        ) -> np.ndarray:
            """
            Define the drift Hamiltonian for a superconducting single qubit.
            :param omega: Characteristic qubit resonant frequency
            :param alpha:
            :param N: The size of the state vector
            :return: A jax array containing the Hamiltonian matrix
            """
            a = qt.destroy(N)

            # H = omega * a.dag() * a + (alpha / 2) * a.dag() * a.dag() * a * a - (omega / 2) * qt.qeye(N)

            H = (alpha / 2) * a.dag() * a.dag() * a * a

            return np.array(H)

        @classmethod
        def gen_control_hamiltonion(cls,
                I: Callable,
                Q: Callable,
                non_linearity: Callable,
                omega: float,
                w_d: float,
                phase: float,
                t: np.ndarray,
                N: int
        ) -> np.ndarray:
            """
            Define the control Hamiltonian for a driven superconducting single qubit system.
            :param I: The in phase pulse envelope
            :param Q: The quadrature pulse envelope
            :param non_linearity: The non-linearity to be applied to the Hamiltonian
            :param w_d: The drive frequency
            :param phase: The phase of the drive signal
            :param t: The time at which to evaluate the Hamiltonian
            :param N: The size of the state vector
            :return: A jax array containing the Hamiltonian matrix
            """
            a = np.array(qt.tensor(qt.destroy(N)))
            a_d = np.array(qt.tensor(qt.destroy(N)).dag())
            # Hs = non_linearity((a+a_d)*(I(t)*np.sin(w_d*t+phase)+Q(t)*np.cos(w_d*t+phase)))

            Hs = non_linearity((a_d*np.exp(1.j*omega*t) + a*np.exp(-1.j*omega*t)) * (I(t) * np.sin(w_d * t + phase) + Q(t) * np.cos(w_d * t + phase)))

            return Hs


class Driven_Single_Qubit_and_Resonator:
    # TODO: implement hamiltonian from https://d-nb.info/1072259729/34 p93 or https://arxiv.org/pdf/2110.05334.pdf eq24 on p9

    @classmethod
    def gen_drift_hamiltonian(cls,
                              omega_q: float,
                              alpha: float,
                              omega_c: float,
                              g: float,
                              N: int
                              ) -> np.ndarray:
        """
        Define the drift Hamiltonian for a superconducting single qubit coupled
        to a resonator in the dispersive regime.
        """
        d = np.matrix(qt.destroy(N))
        eye = np.matrix(qt.qeye(N))

        a = np.kron(d, eye)
        b = np.kron(eye, d)

        H = (omega_q * a.H * a) + ((alpha / 2) * a.H * a.H * a * a) \
            + (g*(b.H*a + b*a.H)) + (omega_c*a.H*a)

        return np.matrix(H)

    @classmethod
    def gen_control_hamiltonion(cls,
                                I: Callable,
                                Q: Callable,
                                non_linearity: Callable,
                                w_d: float,
                                phase: float,
                                lam: float,
                                t: np.ndarray,
                                N: int
                                ) -> np.ndarray:
        """
        Define the control Hamiltonian for a driven superconducting single qubit system.
        :param I: The in phase pulse envelope
        :param Q: The quadrature pulse envelope
        :param non_linearity: The non-linearity to be applied to the Hamiltonian
        :param w_d: The drive frequency
        :param phase: The phase of the drive signal
        :param t: The time at which to evaluate the Hamiltonian
        :param N: The size of the state vector
        :return: A jax array containing the Hamiltonian matrix
        """
        d = np.matrix(qt.destroy(N))
        eye = np.matrix(qt.qeye(N))
        a = np.kron(d, eye)
        b = np.kron(eye, d)

        Hs = non_linearity(((a + a.H) + lam * (b + b.H)) * (I(t) * np.sin(w_d * t + phase) + Q(t) * np.cos(w_d * t + phase)))

        return Hs


def pink_spectrum(frequencies, frequency_cutoff):
    return 1 / (frequencies + frequency_cutoff)

