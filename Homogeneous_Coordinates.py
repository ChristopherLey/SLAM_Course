import numpy as np
import matplotlib.pyplot as plt

from tools import v2t, t2v, parse_landmarks, parse_sensor_data,normalise_angle


def motion_model(x: np.ndarray, u: list[float]) -> np.ndarray:
    x_prime_low = np.array([
        x[0] + u[1] * np.cos(normalise_angle(x[2]) + u[0]),
        x[1] + u[1] * np.sin(normalise_angle(x[2]) + u[0]),
        normalise_angle(x[2]) + u[2] + u[0]])
    x_prime = np.zeros_like(x)
    x_prime[:3] = x_prime_low
    return x_prime


def sensor_model(sensor: list[float]) -> np.ndarray:
    return np.array([
        sensor[0] * np.cos(sensor[1]),
        sensor[0] * np.sin(sensor[1]),
        0.0]
    )


landmarks = parse_landmarks('./data/world.dat')
sensor_data = parse_sensor_data('./data/sensor_data.dat')

# Initial state
x = np.array([0, 0, 0], dtype=float)     # [x, y, theta]

fig, ax = plt.subplots()
plt.ion()
plt.show()

for data in sensor_data:
    ax.cla()
    ax.set_xlim((-2, 12))
    ax.set_ylim((-2, 12))
    for landmark in landmarks:
        circle = plt.Circle((landmark[1], landmark[2]), 0.1, color='k', fill=True)
        ax.add_patch(circle)
    odometry = data.odometry
    x = motion_model(x, odometry)
    print(x)
    robot = plt.Circle((float(x[0]), float(x[1])), 0.1, color='r', fill=False)
    ax.add_patch(robot)
    for sensor in data.sensor_data:
        sensor_absolute = t2v(v2t(x) @ v2t(sensor_model(sensor[1:3])))
        plt.plot([x[0], sensor_absolute[0]], [x[1], sensor_absolute[1]], 'b')
    plt.pause(0.1)

