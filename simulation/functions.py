import jax.numpy as np
from .pauli_matrices import sigma_x, sigma_y, sigma_z

from scipy.linalg import logm

import qutip as qt

def np_map(f, x):
    shape = x.shape
    l = list(map(f, x.flatten()))
    return np.array(l).reshape(*shape, *l[0].shape)

def set_phase(z):
    phi = np.angle(z[..., 0, 0])
    return np.exp(-1j * phi[..., np.newaxis, np.newaxis]) * z

def decompose(U):
    n_sigma = -1j * logm(U)
    P = np.stack([sigma_x, sigma_y, sigma_z], axis=0)
    n = trace(P @ n_sigma)
    # theta = np.linalg.norm(n)
    return n

def to_probability(y):
    return np.abs(y) ** 2

def to_density(y):
    return y @ dag(y)

def expectation(A, y):
    return (dag(y) @ A @ y).real.squeeze()

def trace(A):
    return np.trace(A, axis1=-2, axis2=-1).real

def dag(y):
    return np.swapaxes(y.conj(), axis1=-1, axis2=-2)

def plot_bloch_sphere(states):
    x = expectation(sigma_x, states)
    y = expectation(sigma_y, states)
    z = expectation(sigma_z, states)

    points = np.stack([x, y, z], axis=0)
    b = qt.Bloch()
    b.add_points(points[..., :-1])
    b.add_points(points[..., -1])

    b.render()


def to_real(z):
    return np.concatenate([z.real, z.imag], axis=-1)

def to_complex(x):
    real, imag = np.split(x, 2, axis=-1)
    return real + 1j * imag
