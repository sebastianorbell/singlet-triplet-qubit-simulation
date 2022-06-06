from dataclasses import dataclass
import jax
import diffrax
from scipy.integrate import solve_ivp

from .functions import *
from .pauli_matrices import *

class Qubit:

    def vectorised_matrix_evolve(self, times, control, initial_state, f):
        y = jax.vmap(self.map_matrix_evolve, in_axes=(0, 0, 0, None))(times, control, initial_state, f)
        return y

    @classmethod
    def map_matrix_evolve(cls, times, control, initial_state, f):
        assert times.shape[0] == control.shape[0], f't shape {times.shape}, eps shape {control.shape}'
        control = diffrax.LinearInterpolation(times, control)
        H = lambda t: f(np.array([control.evaluate(t)]))
        func = lambda t, y, args: to_real(-1j * H(t) @ to_complex(y))

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            diffrax.Tsit5(),
            t0=times[0],
            t1=times[-1],
            dt0=(times[1] - times[0]),
            y0=initial_state,
            # stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=times)
        )

        y = np.swapaxes(to_complex(solution.ys), 0, 1)
        return set_phase(y)

    @classmethod
    def evolve(cls, times, control, initial_state, f):

        # H = lambda t: f()
        func = lambda t, y: -1j * f() @ y

        result = solve_ivp(fun=func, y0=initial_state, t_span=(times[0], times[-1]), t_eval=times)
        y = np.swapaxes(result.y, 0, 1)
        return set_phase(y[:, :, np.newaxis])

    # @classmethod
    # def proper_evolve(cls, hamiltonian, times, initial_state):
