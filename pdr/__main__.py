from pdr import estimation
import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'

training_path = os.path.abspath("pdr/data/training")
testing_path = os.path.abspath("pdr/data/testing")

arcore_paths = []
test_arcore_paths = []
imu_paths = []
test_imu_paths = []
for path in os.listdir(training_path):
    if 'IMU' in path:
        imu_paths.append(os.path.join(training_path, path))
    if 'ARCore' in path:
        arcore_paths.append(os.path.join(training_path, path))

for path in os.listdir(testing_path):
    if 'IMU' in path:
        test_imu_paths.append(os.path.join(testing_path, path))
    if 'ARCore' in path:
        test_arcore_paths.append(os.path.join(testing_path, path))

arcore_paths.sort()
test_arcore_paths.sort()
imu_paths.sort()
test_imu_paths.sort()

if __name__ == '__main__':
    os.system('clear')
    estimation.run(arcore_paths, imu_paths, test_arcore_paths, test_imu_paths)
