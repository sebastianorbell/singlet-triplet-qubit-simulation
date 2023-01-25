import jax.numpy as np

ground_projection = np.array([
    [1., 0j],
    [0j, 0j]
])

I = np.array([
    [1., 0.j],
    [0.j, 1.]
])

sigma_x = np.array([
    [0., 1. + 0.j],
    [1. + 0.j, 0.]
])

sigma_y = np.array([
    [0, 0.-1.j],
    [0. + 1.j, 0]
])

sigma_z = np.array([
    [1. + 0.j, 0.],
    [0., -1. + 0.j]
])

sigma_3_z = np.array([
    [1. + 0.j, 0., 0.],
    [0., -1. + 0.j, 0.],
    [0., 0., 0.]
])

r_x = lambda theta: np.array([
    [np.cos(theta/2), -1j * np.sin(theta/2)],
    [-1j*np.sin(theta/2), np.cos(theta/2)]
])

r_y = lambda theta: np.array([
    [np.cos(theta/2), -np.sin(theta/2)],
    [np.sin(theta/2), np.cos(theta/2)]
])

r_z = lambda theta: np.array([
    [np.exp(-1j * theta/2), 0],
    [0, np.exp(1j*theta/2)]
])
