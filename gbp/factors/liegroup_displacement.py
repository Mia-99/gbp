import numpy as np
from scipy.linalg import expm, logm
from utils import transformations, lie_algebra, derivatives

"""
Lie group displacement factor for measurement function h(x_1, x_2) = x_2 * x_1^{-1} and analogous form in higher dimensions.
"""

def meas_fn(x):
    # assert len(x) == 14
    # Rwc1 = transformations.Quaternion(q=x[3:7]).rot_matrix()
    # twc1 = x[:3]
    # Rwc2 = transformations.Quaternion(x[10:14]).rot_matrix()
    # twc2 = x[7:10]
    # Twc1 = np.eye(4)
    # Twc1[:3, :3] = Rwc1
    # Twc1[:3, 3] = twc1
    # Twc2 = np.eye(4)
    # Twc2[:3, :3] = Rwc2
    # Twc2[:3, 3] = twc2
    # return lie_algebra.se3log(np.linalg.inv(Twc1) @ Twc2)
    assert len(x) == 12
    Rwc1 = lie_algebra.so3exp(x[3:6])
    twc1 = x[:3]
    Rwc2 = lie_algebra.so3exp(x[9:12])
    twc2 = x[6:9]
    Twc1 = np.eye(4)
    Twc1[:3, :3] = Rwc1
    Twc1[:3, 3] = twc1
    Twc2 = np.eye(4)
    Twc2[:3, :3] = Rwc2
    Twc2[:3, 3] = twc2
    return lie_algebra.se3log(np.linalg.inv(Twc1) @ Twc2)

def jac_fn(x):
    assert len(x) == 12
    Rwc1 = lie_algebra.so3exp(x[3:6])
    twc1 = x[:3]
    Rwc2 = lie_algebra.so3exp(x[9:12])
    twc2 = x[6:9]

    J = derivatives.jac_fd(x, meas_fn)
    # J[:, :3] = -np.eye(3)
    # J[:, 3:7] = -np.dot(lie_algebra.so3jacinv(Rwc1), twc2 - twc1)
    # J[:, 7:10] = np.eye(3)
    # J[:, 10:14] = np.dot(lie_algebra.so3jacinv(Rwc1), twc2 - twc1)
    return J


if __name__ == '__main__':
    # Check Jacobian function
    x = np.random.rand(12)
    meas_fn(x)
    derivatives.check_jac(jac_fn, x, meas_fn)