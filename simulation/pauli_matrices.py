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