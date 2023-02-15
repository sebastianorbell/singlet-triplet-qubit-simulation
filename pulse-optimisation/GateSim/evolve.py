"""

Created by sebastian.orbell at 14.9.2022

Evolution of a state vector matrix, calculated using the Schrodinger
equation and integrated using Diffrax (a jax based differential equantion
solver). Everything is vectorisable and auto-differentiable.
"""

import jax
import diffrax
from scipy.integrate import solve_ivp
import jax.numpy as np

from functions import *
from pauli_matrices import *

class Evolve:

    @classmethod
    def vectorised_matrix_evolve(self, times, initial_state, f):
        """Vectorised map of the evolution across a set of initial states"""
        y = jax.vmap(self.map_matrix_evolve, in_axes=(None, 0, None))(times, initial_state, f)
        return y

    @classmethod
    def map_matrix_evolve(cls, times: np.ndarray, initial_state: np.ndarray, f: callable):
        """
        Evolution of a state vector with respect to a time dependent hamiltonian,
        using diffrax ODE solvers, which ensures that the whole calculation can
        be automatically differentiated with jax.
        The ODE solver does not work for complex numbers, so I transform to a real
        vector representation of the state vector for the ODE solver,
        and back to the complex state before returning the solution.

        :param times:
        :param initial_state:
        :param f: Callable which accepts a time and returns a Hamiltonian matrix
        :return: ys: The evolution of the state vector
        """

        H = lambda t: f(t)
        func = lambda t, y, args: to_real(-1j * H(t) @ to_complex(y))

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            diffrax.Tsit5(),
            t0=times[0],
            t1=times[-1],
            dt0=(times[1] - times[0]),
            y0=to_real(initial_state),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=times),
            max_steps=16**4
        )

        return to_complex(solution.ys)#set_phase(to_complex(solution.ys))

    @classmethod
    def evolve(cls, times, initial_state, f):
        """Using the scipy ODE solver"""
        func = lambda t, y: -1j * f(t) @ y

        result = solve_ivp(fun=func, y0=initial_state, t_span=(times[0], times[-1]), t_eval=times)
        y = np.swapaxes(result.y, 0, 1)
        return set_phase(y[:, :, np.newaxis])