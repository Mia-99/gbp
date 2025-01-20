import numpy as np
from scipy.linalg import expm, logm

"""
Lie group displacement factor for measurement function h(x_1, x_2) = x_2 * x_1^{-1} and analogous form in higher dimensions.
"""

def meas_fn(x):
    dim = int(len(x) / 2)
    x1 = x[:dim].reshape((dim, dim))
    x2 = x[dim:].reshape((dim, dim))
    return (x2 @ np.linalg.inv(x1)).flatten()

def jac_fn(x):
    dim = int(len(x) / 2)
    x1 = x[:dim].reshape((dim, dim))
    x2 = x[dim:].reshape((dim, dim))
    J1 = -np.kron(np.eye(dim), np.linalg.inv(x1).T)
    J2 = np.kron(np.eye(dim), x2 @ np.linalg.inv(x1).T)
    return np.hstack((J1, J2))
