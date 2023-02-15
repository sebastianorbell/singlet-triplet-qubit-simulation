"""

Created by sebastian.orbell at 8.9.2022

"""
import timeit

import jax

from hamiltonian import Driven_Single_Qubit, return_initial_state
from noise import pink_noise, white_noise
from evolve import Evolve
from signal_processing import *
from pauli_matrices import r_x, sigma_3_z, sigma_z, r_y
from pulse_parameterisation import *
from vector_functions import *
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import jit, vmap, random
from tqdm import tqdm, trange
from functions import plot_bloch_sphere, plot_evolution
import optax
import timeit
from typing import Callable
import numpy
import cma



# from jax.config import config
# config.update("jax_debug_nans", True)

def arctan_non_linearity(x: np.ndarray,
                  alpha: float = 0.63, max_amp: float = 1.) -> np.ndarray:
    """
    Apply a non-linearity to the control Hamiltonian.
    :param x:
    :param alpha:
    :param max_amp:
    :return:
    """
    return max_amp * np.arctan(alpha * x / max_amp) / alpha

def calculate_fidelity(
        pulse_params: np.ndarray,
        pulse_function: Callable,
        initial_states: np.ndarray,
        gate: np.ndarray,
        hd,
        w_d,
        omega,
        phase,
        *args,
        key=0,
        noise_level=1.e-1,
        discretise_function: Callable = lambda f, times, dt: f,
        plot: bool = False,
        error_syndromes: bool = False) -> (float, np.ndarray):

    """
    Calculate the mean state fidelity of a parameterised pulse sequence,
    compared to a target gate, for a set of initial states.

    :param pulse_params: np.ndarray, Parameters defining the pulse envelope.
    :param pulse_function: Callable, Function which parameterises the pulse envelope.
    :param initial_states: np.ndarray, The initial qubit states.
    :param gate: np.ndarray, The target quantum gate.
    :param hd, The drift Hamiltonian of the system
    :param w_d, The drive frequency.
    :param omega, The resonant frequency of the qubit.
    :param phase, The phase of the drive frequency.
    :param *args,
    :param key=0, A key to seed the random sampling of noise.
    :param noise_level=1.e-1, The noise level of the system.
    :param discretise_function: Callable = lambda f, times, dt: f, A function to discretise and filter the pulse envelope.
    :param plot: bool = False A boolean to control the plotting functionality.

    :return The mean infidelity.

    """
    # Calculate the target states
    target_states_sub = vectorised_gate(gate, initial_states[..., :2])
    target_states = np.append(target_states_sub, np.zeros([target_states_sub.__len__(), 1]), axis=-1)

    # Define the I and Q envelopes, as well as a time series for evaluation
    I, Q, times, dt = pulse_function(pulse_params)

    # Discretise and filter pules envelopes
    I = discretise_function(I, times, dt)
    Q = discretise_function(Q, times, dt)

    # Define the non-linearity and construct the control Hamiltonian
    non_linearity = lambda x: x
    keys = jax.random.split(jax.random.PRNGKey(key), 3)
    coloured_noise_sample = noise_level*pink_noise(times.__len__(), keys[0])
    omega_delta = noise_level*white_noise(times.__len__(), keys[1])
    hamiltonian_callable = lambda time: np.interp(time, times, coloured_noise_sample) * sigma_3_z \
                                        + hd + \
                            Driven_Single_Qubit.gen_control_hamiltonion(
                                I, Q, non_linearity, omega + np.interp(time, times, omega_delta), w_d,
                                phase, time, N)

    # hamiltonian_callable = lambda time: hd + Driven_Single_Qubit.gen_control_hamiltonion(
    #                             I, Q, non_linearity, omega, w_d, phase, time, N)

    # Calculate the evolution with respect to the time dependent Hamiltonian,
    # for the set of initial states
    ys = Evolve.vectorised_matrix_evolve(times, initial_states, hamiltonian_callable)

    final_state_vector = ys[:, -1, :]

    # Calculate the loss function
    state_fidelity = return_overlap(final_state_vector, target_states)

    # Plotting utility
    if plot:
        plot_evolution(ys, times, state_fidelity, target_states, I, Q)

    if error_syndromes:
        return return_error_syndromes(final_state_vector, target_states)

    return 1 - np.mean(state_fidelity), final_state_vector

def select_best_initial(gen_params, f, keys):
    """

    A function to evaluate multiple initial guesses of the pulse parameterisation and select the optimum.

    :param gen_params: A function to randomly generate initial guesses.
    :param f: The function to be evaluated.
    :param keys: Keys to seed the random generation of the initial guesses

    :return: The optimial initial guess.

    """
    params = vmap(gen_params)(keys)
    scores = vmap(f, in_axes=0)(params)
    return params[np.argmin(scores)]

def cma_es_optimum(gen_params, f, keys, sigma=1.0, maxfun=50):
    """
    Covariance Matrix Evolution Strategy batch optimisation

    :param gen_params:
    :param f:
    :param keys:
    :param sigma:
    :param maxfun:
    :return:
    """
    initial_params = select_best_initial(gen_params, f, keys)
    es = cma.CMAEvolutionStrategy(initial_params.tolist(), sigma)
    objective = vmap(f)
    count = 0
    while (not es.stop()) and count < maxfun:
        count += 1
        X = es.ask()
        es.tell(X, objective(np.array(X)).tolist())
        es.disp()

    return es.result.xbest

def gradient_optimisation(starting_params, loss_function, N_iters):
    """
    Optimisation via stochastic gradient descent with JAX

    :param starting_params:
    :param loss_function:
    :param N_iters:
    :return:
    """
    start_learning_rate = 1e-1
    optimizer = optax.adam(start_learning_rate)

    # Initialize parameters of the model + optimizer.
    params = np.copy(starting_params)
    opt_state = optimizer.init(params)

    @jit
    def update(params, opt_state):
        """Compute the gradient of the loss function and update the parameters"""
        value, grads = jax.value_and_grad(loss_function)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    losses = []
    # A simple update loop.
    t = trange(N_iters)
    for _ in t:
        params, opt_state, loss = update(params, opt_state)
        losses.append(loss)
        t.set_description('Training loss (loss=%g)' % loss)
    print(f'Optimal params = {params}')

    plt.plot(np.log10(np.array(losses)))
    plt.xlabel('Iteration')
    plt.ylabel('log Infidelity')
    plt.show()

    return params
def initialise_and_optimise(parameter_class, initial_states, gate, hd, w_d, omega, phase,
                            starting_params=None, scale=1.5, max_t=20, noise=1.e-2, N_iters=100, N_guesses=100, N_components=10, N_trials=10):

    """
    Multiple initialisations of the pulse envelope. The optimal initial guess is selected,
    and optimised via gradient descent (this procedure attempts to avoid local optima).
    :param parameter_class:
    :param initial_states:
    :param gate:
    :param hd:
    :param w_d:
    :param omega:
    :param phase:
    :param scale:
    :param max_t:
    :param noise:
    :param N_iters:
    :param N_guesses:
    :param N_components:
    :param N_trials:
    :return:
    """
    key = random.PRNGKey(numpy.random.randint(100))
    pulse_function = lambda x: parameter_class.construct(x, max_t=max_t)
    f_single = lambda x, key: calculate_fidelity(x, pulse_function, initial_states, gate, hd, w_d, omega, phase, key=key, noise_level=noise, discretise_function=discretise_pulse)
    vf = lambda x, key: f_single(x, key)[0]

    mapped = jit(lambda x: np.mean(vmap(vf, in_axes=(None, 0))(x, random.randint(key, [N_trials], 0, 100))))

    gen_params = jit(
        lambda key: parameter_class.random_initial(scale, key=key, N_components=N_components, max_t=max_t))

    # f = jit(lambda x: calculate_fidelity(x, pulse_function, initial_states, gate, hd, w_d, omega, phase, discretise_function=discretise_pulse, noise_level=0.))
    if starting_params is None:
        a = numpy.random.randint(100)
        keys = random.split(random.PRNGKey(a), N_guesses)
        optimal_starting_params = cma_es_optimum(gen_params, mapped, keys)
        # optimal_starting_params = select_best_initial(gen_params, mapped, keys)
        print(f'Optimal starting parameters: {optimal_starting_params}')
        calculate_fidelity(optimal_starting_params, pulse_function, initial_states, gate, hd,w_d, omega, phase, noise_level=noise, discretise_function=discretise_pulse, plot=True)
    else:
        optimal_starting_params=starting_params
        print(f'Optimal starting parameters: {optimal_starting_params}')
        calculate_fidelity(optimal_starting_params, pulse_function, initial_states, gate, hd, w_d, omega, phase,
                           noise_level=noise, discretise_function=discretise_pulse, plot=True)

    params = gradient_optimisation(optimal_starting_params, mapped, N_iters)

    error_syndromes = calculate_fidelity(params, pulse_function, initial_states, gate, hd, w_d, omega, phase, noise_level=noise, discretise_function=discretise_pulse, error_syndromes=True, plot=True)

    fidelity, state_leakage, over_rotation_error, phase_error = error_syndromes

    print(f"Optimal average gate fidelity = {fidelity}")
    print(f"Optimal average state leakage per gate = {state_leakage}")
    print(f"Optimal average over rotation per gate = {over_rotation_error}")
    print(f"Optimal average phase error per gate = {phase_error}")

    return params, fidelity, state_leakage, over_rotation_error, phase_error

if __name__=='__main__':
    N = 3
    omega = 4.53397445414867000 * 2. * np.pi # Typical = 4.5Ghz
    alpha = -0.2040 #* 2. * np.pi # Typical = -0.2Ghz
    w_d = 4.53397445414867000 * 2. * np.pi
    phase = 0.

    g = return_initial_state(0, N)
    e = return_initial_state(1, N)

    initial_states = np.array([
            g,
            return_normalised_state_vector(g+e),
            return_normalised_state_vector(g+1j*e),
            # e
        ])

    hd = Driven_Single_Qubit.gen_drift_hamiltonian(omega, alpha, N)

    theta = np.pi/2.
    gate = r_y(theta)

    parameterisation_options = {
        'Gaussian': Gaussian,
        'DRAG': DRAG,
        'PWC': PWC,
        'Fourier': Fourier,
        'Extended_DRAG': Extended_DRAG,
        'Cosine': Cosine,
        'Extended_cosine': Extended_cosine}

    parameterisation = 'Extended_cosine'

    fidelity_list = []
    state_leakage_list = []

    parameter_class = parameterisation_options[parameterisation]
    t = 10 #ns
    pulse_function = lambda x: parameter_class.construct(x, max_t=t)

    optimal_parameters, fidelity, state_leakage, over_rotation_error, phase_error = initialise_and_optimise(
        parameterisation_options[parameterisation],
        initial_states,
        gate, hd, w_d, omega, phase,
        scale=1e-1,
        noise=1.e-3,
        # N_components=5,
        max_t=t, N_iters=100, N_trials=10)

    fidelity = np.mean(fidelity)
    state_leakage = np.mean(state_leakage)
    print('------------------------------')
    print(f'time = {t}')
    print(f'Fidelity = {fidelity}')
    print(f'State leakage = {state_leakage}')

    fidelity_list.append(fidelity)
    state_leakage_list.append(state_leakage)