import numpy as np
import quaternion

###################################################################################
# Compute world coordinates given relative translation and rotation of the device #
###################################################################################


def world_coordinates_from_rot_trans(init_t, init_q, relative_t, relative_q):
    t = np.array(init_t)
    q = quaternion.from_float_array(init_q)
    pred_t = []
    pred_t.append(t)

    for [delta_t, delta_q] in zip(relative_t, relative_q):
        t = t + np.matmul(quaternion.as_rotation_matrix(q), delta_t.T).T
        q = q * quaternion.from_float_array(delta_q).normalized()
        pred_t.append(np.array(t))
    coord = np.reshape(pred_t, (len(pred_t), 3))
    x = coord[:, 0]
    y = coord[:, 1]
    z = coord[:, 2]
    return (x, y, z)
