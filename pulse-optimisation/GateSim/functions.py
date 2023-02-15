import jax.numpy as np
import matplotlib.pyplot as plt

from pauli_matrices import sigma_x, sigma_y, sigma_z
from vector_functions import vectorised_expectation
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

def plot_bloch_sphere(states, target, fig, ax, text=None):
    x = vectorised_expectation(sigma_x, states)
    y = vectorised_expectation(sigma_y, states)
    z = vectorised_expectation(sigma_z, states)

    points = np.stack([x, y, z])


    x_t = vectorised_expectation(sigma_x, np.array([target]))
    y_t = vectorised_expectation(sigma_y, np.array([target]))
    z_t = vectorised_expectation(sigma_z, np.array([target]))

    points_t = np.stack([x_t, y_t, z_t])

    b = qt.Bloch(fig=fig, axes=ax)
    b.add_points(points.tolist(), meth='l')
    b.add_points(points_t.tolist(), meth='s')
    b.render()

    if text:
        b.add_annotation([0,0,1], text)
    # b.show()

def plot_evolution(ys, times, state_fidelity, target_states, I, Q):
    n_states = target_states.__len__()
    for index, state in enumerate(ys):
        axn = plt.subplot(2, n_states, (index + 1, index + 1))
        axn.plot(times, np.square(np.abs(np.squeeze(state))), label=['g', 'e', 'f'])
        axn.scatter(np.full_like(target_states[index, :], times[-1]), np.square(np.abs(target_states[index, :])),
                    c=['#1f77b4', '#ff7f0e', '#2ca02c'], marker='*', alpha=0.5)
        axn.legend()
        axn.set_xlabel('Time (ns)')
        axn.set_ylabel('State probability')

        axn.set_title(f'Fidelity = {state_fidelity[index]:1.4f}')

    axn = plt.subplot(2, n_states, (n_states + 1, 2 * n_states))
    axn.plot(times, I(times), '--', label='I')
    axn.plot(times, Q(times), '--', label='Q')
    axn.legend()
    axn.set_xlabel('Time (ns)')
    axn.set_ylabel('Pulse envelope amplitude')
    plt.tight_layout()
    plt.show()

    target_states_sub = target_states[..., :-1]
    fig = plt.figure(figsize=plt.figaspect(0.5))
    for index, (states, target, fidelity) in enumerate(zip(ys[..., :-1], target_states_sub, state_fidelity)):
        ax = fig.add_subplot(1, n_states, index + 1, projection='3d')
        plot_bloch_sphere(states, target, fig, ax, text=f'Fidelity = {fidelity:1.4f}')
        ax.set_title(f'Fidelity = {fidelity:1.4f}')
    plt.tight_layout()
    plt.show()

def to_real(z):
    return np.concatenate([z.real, z.imag], axis=-1)

def to_complex(x):
    real, imag = np.split(x, 2, axis=-1)
    return real + 1j * imag

if __name__=='__main__':
    from hamiltonian import return_initial_state
    from vector_functions import return_normalised_state_vector, vectorised_gate, vectorised_expectation
    from pauli_matrices import r_y, r_x

    import matplotlib.pyplot as plt
    N = 2
    g = return_initial_state(0, N)
    e = return_initial_state(1, N)
    initial_states = np.array([
            g,
            return_normalised_state_vector(g+e),
            return_normalised_state_vector(g+1j*e),
            # e
        ])

    initial_x = vectorised_expectation(sigma_x, initial_states)
    initial_y = vectorised_expectation(sigma_y, initial_states)
    initial_z = vectorised_expectation(sigma_z, initial_states)

    i_vecs = np.stack([initial_x, initial_y, initial_z]).T
    vec = [0, 1, 0]

    theta = np.pi
    gate = r_x(theta)
    target_states = vectorised_gate(gate, initial_states)

    target_x = vectorised_expectation(sigma_x, target_states)
    target_y = vectorised_expectation(sigma_y, target_states)
    target_z = vectorised_expectation(sigma_z, target_states)

    t_vecs = np.stack([target_x, target_y, target_z]).T

    # b = qt.Bloch()
    # b.add_vectors(i_vecs.tolist()[0])
    # b.add_vectors(t_vecs.tolist()[0])
    # b.render()
    # b.show()
    #
    # b = qt.Bloch()
    # b.add_vectors(i_vecs.tolist()[1])
    # b.add_vectors(t_vecs.tolist()[1])
    # b.render()
    # b.show()
    #
    # b = qt.Bloch()
    # b.add_vectors(i_vecs.tolist()[2])
    # b.add_vectors(t_vecs.tolist()[2])
    # b.render()
    # b.show()
    #
    # th = np.linspace(0, 2 * np.pi, 20)
    # xp = np.cos(th)
    # yp = np.sin(th)
    # zp = np.zeros(20)
    #
    # pnts = np.array([xp, yp, zp])
    # b.add_points(pnts.tolist())
    # b.render()
    # b.show()


