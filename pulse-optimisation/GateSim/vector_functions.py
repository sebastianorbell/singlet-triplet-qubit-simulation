"""

Created by sebastian.orbell at 12.9.2022

"""

import jax.numpy as np
from jax import vmap

def return_normalised_state_vector(state):
    """Normalise a state vector"""
    return state/np.sqrt(np.sum(np.square(np.abs(state))))

def return_overlap(output_states, target_states):
    """Calculate the vector inner product of a set of output states to a set of target states"""
    # state_fidelities = vmap(lambda a, b: 1. - 0.5*np.sqrt(np.sum(np.square(np.abs(a - b)))), in_axes=(0, 0))(target_states, output_states)
    output_norm = vmap(lambda a: a/np.sqrt(np.sum(np.square(np.abs(a)))))(output_states)
    state_fidelities = vmap(lambda a, b: np.square(np.abs(np.matmul(a.conj().T, b))), in_axes=(0, 0))(target_states, output_norm)
    return state_fidelities

def return_error_syndromes(output_states, target_states):
    """Calculate the vector inner product of a set of output states to a set of target states"""
    output_norm = vmap(lambda a: a/np.sqrt(np.sum(np.square(np.abs(a)))))(output_states)
    state_fidelities = vmap(lambda a, b: np.square(np.abs(np.matmul(a.conj().T, b))), in_axes=(0, 0))(target_states, output_norm)
    state_leakage = vmap(lambda a : np.square(np.abs(a[-1])), in_axes=(0))(output_norm)
    over_rotation_error = vmap(lambda a, b: np.abs(np.sum(a[:-1] - b[:-1])), in_axes=(0, 0))(target_states, output_norm)
    phase_error = vmap(lambda a, b: np.angle(np.sum(a[:-1] - b[:-1])), in_axes=(0, 0))(target_states, output_norm)
    return state_fidelities, state_leakage, over_rotation_error, phase_error

def vectorised_gate(gate, states):
    """Vectorise the matrix multiplication of a gate across a set of initial states"""
    return vmap(lambda g, s: np.matmul(g, s), in_axes=(None, 0))(gate,  states)

def observable(O, state_vector):
    return np.real(np.dot(np.dot(state_vector.conjugate(), O), state_vector))

def vectorised_expectation(operator, states):
    return vmap(observable, in_axes=(None, 0))(operator, states)