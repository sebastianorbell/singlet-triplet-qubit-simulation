"""

Created by sebastian.orbell at 7.11.2022

"""

import enum

import numpy as np


@enum.unique
class XYGate(enum.IntEnum):
    """Enumerates all single-qubit Clifford gates whose rotation axis is in the XY plane.

    Members of this enum can be mapped to the corresponding unitary propagator using
    the dictionary returned by :meth:`XYGate.get_unitaries`,
    or to the pulse implementing the gate using :meth:`.RotationPulseFactory.__getitem__`.

    Only used in the tomography experiments.
    """

    IDENTITY = 0
    X_90 = 1
    X_180 = 2
    X_M90 = 3
    Y_90 = 4
    Y_180 = 5
    Y_M90 = 6
    X_90_custom = 7


XYGATE_UNITARIES = {
    XYGate.IDENTITY: np.eye(2, dtype=complex),
    XYGate.X_90: np.array([[1, -1j], [-1j, 1]], dtype=complex) / np.sqrt(2),
    XYGate.X_180: np.array([[0, -1j], [-1j, 0]], dtype=complex),
    XYGate.X_M90: np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2),
    XYGate.Y_90: np.array([[1, -1], [1, 1]], dtype=complex) / np.sqrt(2),
    XYGate.Y_180: np.array([[0, -1], [1, 0]], dtype=complex),
    XYGate.Y_M90: np.array([[1, 1], [-1, 1]], dtype=complex) / np.sqrt(2),
    XYGate.X_90_custom: np.array([[1, -1j], [-1j, 1]], dtype=complex) / np.sqrt(2),
}
"""Mapping of XYGates to the corresponding SU(2) matrices"""
