import numpy as np


def v2t(v: np.ndarray) -> np.ndarray:
    """
    Convert a 3-vector to a 3x3 homogeneous transformation matrix
    """
    t = np.eye(3)
    t[0, :] = [np.cos(v[2]), -np.sin(v[2]), v[0]]
    t[1, :] = [np.sin(v[2]), np.cos(v[2]), v[1]]
    return t


def t2v(t: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 homogeneous transformation matrix to a 3-vector
    """
    v = np.zeros(3)
    v[0] = t[0, 2]
    v[1] = t[1, 2]
    v[2] = np.arctan2(t[1, 0], t[0, 0])
    return v
