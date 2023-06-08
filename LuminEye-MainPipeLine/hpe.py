import numpy as np

zprev = np.zeros((68,68))
r0 = np.zeros((3,3))


def compute_rotation(shape):

    w = measurement_matrix(shape)

    r = orthogonal_iter(w[[0], :], w[[1], :], 3)

    return r


def measurement_matrix(shape):

    u = shape[:, [0]].T
    v = shape[:, [1]].T

    u_mean = np.mean(u)
    u = u - u_mean

    v_mean = np.mean(v)
    v = v - v_mean

    w = np.array([u, v]).reshape(2, 68)

    # print(w, '\n')

    return w


def orthogonal_iter(x, y, r):

    global zprev

    # Compute the covariance-type matrix, Z
    z = zprev + np.dot(x.T, x) + np.dot(y.T, y)

    # Compute the motion and shape components
    v = np.linalg.svd(z)[0]
    qbar = v[:, 0:r]

    # Compute the matrix containing the pose and configuration weights
    q = np.array([np.matmul(x, qbar), np.matmul(y, qbar)]).reshape(1, 2 * r)

    # Factor matrix q into the pose, r_hat, and the configuration weights, l
    l, r_hat = q_factorisation(q)

    # Extract the pose vectors
    i = r_hat[:, 0:3]
    j = r_hat[:, 3:6]

    # Impose orthonormality
    vects = np.concatenate((i, j)).T
    v_orth, sigma_orth, u_orth = np.linalg.svd(vects, full_matrices=False)
    r_orth = np.matmul(u_orth, v_orth.T)

    cross_vects = np.cross(r_orth[0, :], r_orth[1, :])
    r_orth = np.vstack([r_orth, cross_vects])

    zprev = z

    return r_orth


def q_factorisation(q):

    u, sigma, vh = np.linalg.svd(q)

    r = u * np.sqrt(sigma)
    s = np.sqrt(sigma) * vh[[0], :]

    return r, s


def compute_angles(r, counter):

    yaw = pitch = roll = yaw_deg = pitch_deg = roll_deg = 0

    global r0

    # Align with the first camera reference frame
    if counter == 10:

        i = r[0, :] / np.linalg.norm(r[0, :])
        j = r[1, :] / np.linalg.norm(r[1, :])
        k = np.cross(i, j)

        # Create rotation matrix from normalised vectors
        r0 = np.array([i, j, k]).T

    if counter >= 10:

        r_new = np.matmul(r, r0)

        i_new = r_new[0, :] / np.linalg.norm(r_new[0, :])
        j_new = r_new[1, :] / np.linalg.norm(r_new[1, :])
        k_new = np.cross(i_new, j_new)

        # Create rotation matrix from normalised vectors
        r_vects = np.array([i_new, j_new, k_new]).T

        if counter == 10:
            print(i_new, j_new, k_new, '\n')
            print(r_vects, '\n')

        yaw = -np.arcsin(r_vects[0, 2])
        pitch = np.arcsin(r_vects[1, 2] / np.cos(yaw))
        roll = np.arccos(r_vects[0, 0] / np.cos(yaw))

        yaw_deg = yaw * (180 / np.pi)
        pitch_deg = pitch * (180 / np.pi)
        roll_deg = roll * (180 / np.pi)

        # print(yaw, pitch, roll, '\n')

    return yaw, pitch, roll, yaw_deg, pitch_deg, roll_deg
