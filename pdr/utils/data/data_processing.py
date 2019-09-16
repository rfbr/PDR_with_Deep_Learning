#########################################################################
#                                                                       #
#             Define data preprocessing for 6DOF estimation             #
#                                                                       #
#########################################################################
import numpy as np
from pdr.utils.geometry.relative_coordinates import relative_rotation_translation
from tqdm import tqdm


def gravity_filter(accelerations):
    alpha = 0.8
    linear_accelerations = []
    gravity = [0, 0, 0]
    for acc in accelerations:
        gravity[0] = alpha * gravity[0] + (1 - alpha) * acc[0]
        gravity[1] = alpha * gravity[1] + (1 - alpha) * acc[1]
        gravity[2] = alpha * gravity[2] + (1 - alpha) * acc[2]
        linear_acceleration = [0, 0, 0]
        linear_acceleration[0] = acc[0] - gravity[0]
        linear_acceleration[1] = acc[1] - gravity[1]
        linear_acceleration[2] = acc[2] - gravity[2]
        linear_accelerations.append(linear_acceleration)
    return linear_accelerations

####################################################################################################################
# Return all posible relative translation and quaternion rotation between 2 positions possible (distant by 'step') #
# and 'lookback' number of past IMU data.                                                                          #
####################################################################################################################


def preprocessing(ARCore, IMU, lookback, step):

    acc_samples = []
    gyr_samples = []
    t_targets = []
    q_targets = []

    for i in tqdm(range(len(ARCore)), desc='Preprocessing data'):
        IMU[i][:, 1:4] = gravity_filter(IMU[i][:, 1:4])
        # Find the first ARCore index such as we have 'lookback' IMU data
        k = np.where(ARCore[i][:, 0] >
                     IMU[i][lookback-1, 0])[0][0]
        for j in range(max(k-step, 0), max(k-step, 0)+step):
            *_, relative_t, relative_q = relative_rotation_translation(
                ARCore[i][j::step, 1:4], ARCore[i][j::step, 4:8])
            n = len(relative_t)
            t_targets.append(relative_t)
            q_targets.append(relative_q)
            acc_sample = np.zeros((n, lookback, 3))
            gyr_sample = np.zeros((n, lookback, 3))
            for l in tqdm(range(n)):
                index = np.where(IMU[i][:, 0] < ARCore[i]
                                 [(1+l)*step+j, 0])[0][-lookback:]
                acc_sample[l] = IMU[i][index, 1:4]
                gyr_sample[l] = IMU[i][index, 4:]
            acc_samples.append(acc_sample)
            gyr_samples.append(gyr_sample)

    acc_samples = np.vstack(acc_samples)
    gyr_samples = np.vstack(gyr_samples)
    t_targets = np.vstack(t_targets)
    q_targets = np.vstack(q_targets)

    return (acc_samples, gyr_samples), (t_targets, q_targets)

######################################################################################################################################
# Use linear interpolation and spherical linear interpolation (slerp) to compute translation and rotation according to IMU timestamp #
######################################################################################################################################


def augmented_preprocessing(augmented_data, lookback, step):

    acc_samples = []
    gyr_samples = []
    t_targets = []
    q_targets = []

    for i in tqdm(range(len(augmented_data))):
        augmented_data[i][:, 1:4] = gravity_filter(augmented_data[i][:, 1:4])
        for j in range(max(lookback-step, 0), max(lookback-step, 0)+step):
            *_, relative_t, relative_q = relative_rotation_translation(
                augmented_data[i][j::step, 7:10], augmented_data[i][j::step, 10:])
            n = len(relative_t)
            t_targets.append(relative_t)
            q_targets.append(relative_q)
            acc_sample = np.zeros((n, lookback, 3))
            gyr_sample = np.zeros((n, lookback, 3))
            for l in tqdm(range(n)):
                index = range(j+step*(1+l)-lookback, j+step*(l+1))
                acc_sample[l] = augmented_data[i][index, 1:4]
                gyr_sample[l] = augmented_data[i][index, 4:7]
            acc_samples.append(acc_sample)
            gyr_samples.append(gyr_sample)

    acc_samples = np.vstack(acc_samples)
    gyr_samples = np.vstack(gyr_samples)
    t_targets = np.vstack(t_targets)
    q_targets = np.vstack(q_targets)

    return (acc_samples, gyr_samples), (t_targets, q_targets)


###############################################################################################
# Return relative translation and quaternion rotation between 2 positions (distant by 'step') #
# and 'lookback' number of past IMU data for testing data                                     #
###############################################################################################
def predictions_processing(ARCore, IMU, lookback, step):

    acc_samples = []
    gyr_samples = []
    t_targets = []
    q_targets = []
    init_ts = []
    init_qs = []

    for i in range(len(ARCore)):
        IMU[i][:, 1:4] = gravity_filter(IMU[i][:, 1:4])
        k = np.where(ARCore[i][:, 0] >
                     IMU[i][lookback-1, 0])[0][0]
        init_t, init_q, relative_t, relative_q = relative_rotation_translation(
            ARCore[i][max(k-step, 0)::step, 1:4], ARCore[i][max(k-step, 0)::step, 4:])
        init_ts.append(init_t)
        init_qs.append(init_q)
        n = len(relative_t)

        t_targets.append(relative_t)
        q_targets.append(relative_q)

        acc_sample = np.zeros((n, lookback, 3))
        gyr_sample = np.zeros((n, lookback, 3))

        for l in range(n):
            index = np.where(IMU[i][:, 0] < ARCore[i]
                             [l*step+max(k, step), 0])[0][-lookback:]
            acc_sample[l] = IMU[i][index, 1:4]
            gyr_sample[l] = IMU[i][index, 4:7]
        acc_samples.append(acc_sample)
        gyr_samples.append(gyr_sample)

    return init_ts, init_qs, (acc_samples, gyr_samples), (t_targets, q_targets)

######################################################################################################################################
# Use linear interpolation and spherical linear interpolation (slerp) to compute translation and rotation according to IMU timestamp #
######################################################################################################################################


def augmented_predictions_processing(augmented_data, lookback, step):

    acc_samples = []
    gyr_samples = []
    t_targets = []
    q_targets = []
    init_ts = []
    init_qs = []

    for i in range(len(augmented_data)):
        augmented_data[i][:, 1:4] = gravity_filter(augmented_data[i][:, 1:4])
        init_t, init_q, relative_t, relative_q = relative_rotation_translation(augmented_data[i][max(
            lookback-step, 0)::step, 7:10], augmented_data[i][max(lookback-step, 0)::step, 10:])
        init_ts.append(init_t)
        init_qs.append(init_q)
        n = len(relative_t)

        t_targets.append(relative_t)
        q_targets.append(relative_q)

        acc_sample = np.zeros((n, lookback, 3))
        gyr_sample = np.zeros((n, lookback, 3))

        for l in range(n):
            index = range(max(0, step-lookback)+step *
                          l, max(lookback, step)+step*l)
            acc_sample[l] = augmented_data[i][index, 1:4]
            gyr_sample[l] = augmented_data[i][index, 4:7]
        acc_samples.append(acc_sample)
        gyr_samples.append(gyr_sample)

    return init_ts, init_qs, (acc_samples, gyr_samples), (t_targets, q_targets)
