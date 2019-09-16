import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

##################################################################
# Use linear interpolation to create new data from existing ones #
##################################################################


def linear_interpolation(ARCore, IMU):

    augmented_data = []

    for arcore, imu in tqdm(zip(ARCore, IMU), desc="Processing data augmentation"):
        begin = np.where(imu[:, 0] >= arcore[0, 0])[0][0]
        end = np.where(imu[:, 0] <= arcore[-1, 0])[0][-1] + 1
        timestamp = np.reshape(imu[begin:end, 0], (end-begin, 1))
        x, y, z = map(lambda x: np.interp(timestamp, arcore[:, 0], x), [
                      arcore[:, i] for i in range(1, 4)])
        key_rots = R.from_quat(arcore[:, [5, 6, 7, 4]])
        key_times = arcore[:, 0]
        slerp = Slerp(key_times, key_rots)
        interp_rots = slerp(imu[begin:end, 0]).as_quat()[:, [3, 0, 1, 2]]
        data = np.concatenate(
            (timestamp, imu[begin:end, 1:], x, y, z, interp_rots), axis=1)
        augmented_data.append(data)

    return augmented_data
