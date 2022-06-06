import numpy as np
from dataclasses import dataclass
from typing import Union, List
# import torch

class Gate:

    def __init__(self, durations: np.ndarray, epsilon: np.ndarray):
        self.durations = durations if isinstance(durations, np.ndarray) else np.array([durations])
        self.epsilon = epsilon if isinstance(epsilon, np.ndarray) else np.array([epsilon])
        self.times = np.cumsum(durations)

    def __iter__(self):
        return zip(reversed(self.durations), reversed(self.epsilon))

    def __call__(self, t):
        id = np.searchsorted(self.times, t, side='left')
        return self.epsilon[id]

    def __repr__(self):
        return 'durations: {}\n' \
               'epsilon: {}'.format(self.durations, self.epsilon)

    def __add__(self, other):
        return Gate(
            durations=np.concatenate([self.durations, other.durations]),
            epsilon=np.concatenate([self.epsilon, other.epsilon])
        )

    def min(self):
        return 0.

    def max(self):
        return self.times.max()


class ControlField:

    def __init__(self, times: np.ndarray, epsilon: np.ndarray):

        self.times = times if isinstance(times, np.ndarray) else np.array([times])
        self.epsilon = epsilon if isinstance(epsilon, np.ndarray) else np.array([epsilon])

        assert self.times.shape == self.epsilon.shape, f'{self.times.shape}, {self.epsilon.shape}'

    def __iter__(self):
        return zip(reversed(self.times), reversed(self.epsilon))

    def __call__(self, t):
        id = np.searchsorted(self.times, t, side='left')
        return self.epsilon[id-1]

    def __repr__(self):
        return 'times: {}\n' \
               'epsilon: {}'.format(self.times, self.epsilon)

    def __add__(self, other):
        return ControlField(
            times=np.concatenate([self.times, self.times[-1] + other.durations]),
            epsilon=np.concatenate([self.epsilon, other.epsilon])
        )

    def min(self):
        return 0.

    def max(self):
        return self.times.max()
