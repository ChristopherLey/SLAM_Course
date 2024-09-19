import numpy as np


def normalise_angle(angle: float | np.ndarray) -> float:
    """
    Normalise an angle to be between -pi and pi
    """
    return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))


def rad2deg(angle: float | np.ndarray) -> float:
    """
    Convert an angle from radians to degrees
    """
    return angle * 180.0 / np.pi


def deg2rad(angle: float | np.ndarray) -> float:
    """
    Convert an angle from degrees to radians
    """
    return angle * np.pi / 180.0


