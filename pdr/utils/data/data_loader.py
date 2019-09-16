import pandas as pd
from tqdm import tqdm

#################################
# Load and format recorded data #
#################################


def data_loader(paths, data_type):
    if(data_type not in ('arcore', 'imu')):
        print('{} not supported'.format(data_type))
        return
    data = []
    for path in tqdm(paths, desc="Loading ARCore data" if data_type == 'arcore' else "Loading IMU data"):
        temp = pd.read_csv(path).values
        if(data_type == 'arcore'):
            # Change coordinate system to have x fordward the pedestrian, y on his left, change quaternion order (w,x,y,z)
            temp = temp[:, [0, 3, 1, 2, 7, 6, 4, 5]]
            temp[:, [1, 2, 5, 6]] *= -1
            # Since the writting is multi-threaded sometimes the timestamps are not in chronological order
            temp = temp[temp[:, 0].argsort()]
        else:
            temp = temp[:, [0, 3, 1, 2, 6, 4, 5]]
            temp[:, [1, 2, 4, 5]] *= -1
            temp = temp[temp[:, 0].argsort()]
        data.append(temp)
    return data
