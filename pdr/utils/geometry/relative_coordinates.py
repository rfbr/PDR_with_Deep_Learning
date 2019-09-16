import numpy as np
import quaternion


def force_quaternion_uniqueness(q):

    q_data = quaternion.as_float_array(q)

    if np.absolute(q_data[0]) > 1e-05:
        if q_data[0] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[1]) > 1e-05:
        if q_data[1] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[2]) > 1e-05:
        if q_data[2] < 0:
            return -q
        else:
            return q
    else:
        if q_data[3] < 0:
            return -q
        else:
            return q

###################################################################################
# Compute relative translation and rotation given global translation and rotation #                                                               #
###################################################################################


def relative_rotation_translation(translation, rotation):

    n = len(translation)-1
    relative_t = np.zeros((n, 3))
    relative_q = np.zeros((n, 4))

    init_t = translation[0]
    init_q = rotation[0]
    for i in range(n):
        t_a = translation[i]
        t_b = translation[i+1]
        q_a = quaternion.from_float_array(rotation[i])
        q_b = quaternion.from_float_array(rotation[i+1])

        relative_t[i] = np.matmul(
            quaternion.as_rotation_matrix(q_a).T, (t_b.T-t_a.T)).T
        relative_q[i] = quaternion.as_float_array(
            force_quaternion_uniqueness(q_a.conjugate()*q_b))

    return init_t, init_q, relative_t, relative_q
