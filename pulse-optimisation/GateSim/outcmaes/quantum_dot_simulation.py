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

