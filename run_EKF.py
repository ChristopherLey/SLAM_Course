import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

from tools import v2t, t2v, parse_landmarks, parse_sensor_data, normalise_angle


def prediction_step(mu, sigma, u):
    R_robot = np.eye(3) * 0.1
    R_robot[2, 2] = 0.1 / 10
    R = np.zeros_like(sigma)
    R[:3, :3] = R_robot
    u_r1, u_t, u_r2 = u

    mu[0, 0] += u_t * np.cos(mu[2].item() + u_r1)
    mu[1, 0] += u_t * np.sin(mu[2].item() + u_r1)
    mu[2, 0] += u_r1 + u_r2
    mu[2, 0] = normalise_angle(mu[2, 0])

    G_x = np.eye(3)
    G_x[0, 2] = -u_t * np.sin(mu[2].item() + u_r1)
    G_x[1, 2] = u_t * np.cos(mu[2].item() + u_r1)
    G = np.eye(mu.shape[0])
    G[:3, :3] = G_x

    sigma = G @ sigma @ G.T + R
    return mu, sigma


def correction_step(mu, sigma, sensor, observed_landmarks):
    m = len(sensor)
    dim = mu.shape[0]
    z = np.zeros((m*2, 1), dtype=float)
    expected_z = np.zeros((m*2, 1), dtype=float)

    H = []

    for j, (landmark_id, r, phi) in enumerate(sensor):
        landmark_id = int(landmark_id) - 1
        if not observed_landmarks[landmark_id]:
            mu[2 * landmark_id + 3] = mu[0] + r * np.cos(mu[2] + phi)
            mu[2 * landmark_id + 4] = mu[1] + r * np.sin(mu[2] + phi)
            observed_landmarks[landmark_id] = True
        z[2 * j] = r
        z[2 * j + 1] = phi
        delta = mu[2 * landmark_id + 3: 2 * landmark_id + 5] - mu[0:2]
        q = delta.T @ delta

        expected_z[2 * j] = np.sqrt(q)
        expected_z[2 * j + 1] = normalise_angle(np.arctan2(delta[1], delta[0]) - mu[2])

        H_i = np.block([ # noqa
            [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], np.zeros((1, 1)), np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
            [delta[1:2], -delta[0:1], -q, -delta[1:2], delta[0:1]]
        ]) * 1/q

        F = np.zeros((5, dim), dtype=float)
        F[:3, :3] = np.eye(3)
        F[3:, 2 * landmark_id + 3: 2 * landmark_id + 5] = np.eye(2)

        H.append(H_i @ F)

    H = np.concatenate(H, axis=0)
    Q = np.eye(2*m) * 0.01

    K = sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Q)

    z_diff = z - expected_z
    z_diff[1::2] = normalise_angle(z_diff[1::2])

    mu = mu + K @ z_diff
    mu[2] = normalise_angle(mu[2])
    sigma = (np.eye(dim) - K @ H) @ sigma

    return mu, sigma, observed_landmarks


landmarks = parse_landmarks('./data/world.dat')
N = len(landmarks)
sensor_data = parse_sensor_data('./data/sensor_data.dat')

# Initialise EKF
mu = np.zeros((2 * N + 3, 1), dtype=float)
robot_sigma = np.zeros((3, 3), dtype=float)
robot_map_sigma = np.zeros((3, 2 * N), dtype=float)
map_sigma = 1000.0 * np.eye(2 * N)
sigma = np.block([[robot_sigma, robot_map_sigma], [robot_map_sigma.T, map_sigma]]) # noqa

# Q = np.eye(2) * 0.01
observed_landmarks = np.zeros(N, dtype=bool)

fig, ax = plt.subplots(figsize=(6, 6))
plt.setp(ax, xlim=(-2, 12), ylim=(-2, 12), aspect=1)
plt.ion()
plt.tight_layout()
plt.show()


for data in sensor_data:
    ax.cla()
    ax.set_xlim((-2, 12))
    ax.set_ylim((-2, 12))
    for landmark in landmarks:
        circle = Circle((landmark[1], landmark[2]), 0.1, color='k', fill=True)
        ax.add_patch(circle)

    # EKF Inference
    mu, sigma = prediction_step(mu, sigma, data.odometry)
    mu, sigma, observed_landmarks = correction_step(mu, sigma, data.sensor_data, observed_landmarks)

    std_x = np.sqrt(sigma[0, 0])
    std_y = np.sqrt(sigma[1, 1])
    robot = Circle((mu[0].item(), mu[1].item()), 0.1, color='r', fill=False)
    robot_uncertainty = Ellipse((mu[0].item(), mu[1].item()), width=2 * std_x, height=2 * std_y, color='r', fill=False)
    ax.add_patch(robot)
    ax.add_patch(robot_uncertainty)
    for i in range(N):
        if observed_landmarks[i]:
            mu_jx, mu_jy = mu[2 * i + 3].item(), mu[2 * i + 4].item()
            landmark_estimation_mu = plt.Circle((mu[2 * i + 3].item(), mu[2 * i + 4].item()), 0.1, color='b', fill=False)
            ax.add_patch(landmark_estimation_mu)
            std_x = np.sqrt(sigma[2 * i + 3, 2 * i + 3])
            std_y = np.sqrt(sigma[2 * i + 4, 2 * i + 4])
            landmark_uncertainty = Ellipse((mu[2 * i + 3].item(), mu[2 * i + 4].item()), width=2*std_x, height=2*std_y, color='b', fill=False)
            ax.add_patch(landmark_uncertainty)
            for landmark_id, r, phi in data.sensor_data:
                if int(landmark_id) - 1 == i:
                    z_observed_x = mu[0] + r * np.cos(mu[2] + phi)
                    z_observed_y = mu[1] + r * np.sin(mu[2] + phi)
                    ax.plot([mu[0].item(), z_observed_x.item()], [mu[1].item(), z_observed_y.item()], 'b', alpha=0.1)
    plt.pause(0.1)
