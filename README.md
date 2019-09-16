# Deep learning based pedestrian dead reckoning using smartphone IMU

Introduction
=
Selfsufficient running and independance from the outside environment give smarthpone built-in inertial sensors an important role in indoor navigation. This repository allows the creation of a mobile real-time indoor positioning system using a smart-phone held in hand. 
In order to achieve this, we will only use the Inertial Measurement Unit (IMU) present in the phone, gathering acceleration and angular velocity. The proposed method is based on 6-DOF odometry, computing relative sensor pose changes between two sequential moments using a neural network. The estimate will then be refined by a Bayesian filter, a particle filter, associated with a map-matching algorithm.

 This repository contains only the 6-DOF odometry implmentation using IMU data gathered with an android application, powered by a CNN-LSTM neural network. It allows to create a tflite model with the best trained model, ready to be use in our android application.

Global trajectory estimation
=
To estimate the pedestrian 3D world coordinates, we use our neural network to predict 6DOF relative coordinates (represented by a translation vector and a quaternion rotation vector) between two given timestamps separated by a constant step.
Thus, by combining all relative poses, we can compute the global pose.

Data format
=
We use IMU data (acceleration and angular velocity) gathered by an android application via embedded accelerometer and gyroscope to estimate the ground truth position captured via ARCore.

The neural network
=
Our neural network can be divided into three main parts:
- A convolutional part: accelerometer and gyroscope data are preprocessed separatly. We use 3 layers to extract local features (1D convolutional layer/max pooling/1D convolutional layer) and then concatenate the results.
- A LSTM part: we use two LSTM layers to process preprocessed IMU data as a time series.
- A multi-loss layer: this part is used to optimize the different weights between the different loss functions used.

Project architecture
=
- Data folder: it contains tensorboard logs, saved models by Keras callback and training and testind data.
- Utils folder: it contains data processing, geometry,models and plotting functions.
- Estimation files: there are the main files used to load the data, preprocess them, train the neural network, predict the relative poses and plot the global estimated trajectory.

Prerequisite
=
- Nvidia driver: 410.104
- CUDA version: 10.0
- Python: 3.7
- Tensorflow-gpu: 1.14.0

Usage 
= 
I recommend you use Anaconda distribution to create a virtual environment as follow:
```console
$ conda create -n tf_env python=3.7
$ conda activate tf_env
$ pip install tensorflow-gpu
$ pip install numpy-quaternion
$ pip install tfquaternion
$ pip install tqdm
$ conda install scikit-learn
$ conda install basemap
$ conda install pandas
$ conda install numba
```
This project was designed to be user friendly, just execute ```python -m pdr``` and follow the different instructions.

Results
= 
You can find some results in the img folder. Here are some examples:
- With normal training data:
![alt text](https://github.com/rfbr/PDR/tree/master/img/1.png)
![alt text](https://github.com/rfbr/PDR/tree/master/img/2.png)
![alt text](https://github.com/rfbr/PDR/tree/master/img/4.png)
- With augmented training data (linear interpolation):
![alt text](https://github.com/rfbr/PDR/tree/master/img/Figure_1.png)
![alt text](https://github.com/rfbr/PDR/tree/master/img/Figure_2.png)
![alt text](https://github.com/rfbr/PDR/tree/master/img/Figure_4.png)

Concerning floor changing detection, data augmentation leads to bad results. Spherical linear interpolation for quaternions might not be the best or we may have done some mistakes: 
- With normal training data:
![alt text](https://github.com/rfbr/PDR/tree/master/img/6.png)
![alt text](https://github.com/rfbr/PDR/tree/master/img/7.png)

- With augmented training data (linear interpolation):
![alt text](https://github.com/rfbr/PDR/tree/master/img/Figure_6.png)
![alt text](https://github.com/rfbr/PDR/tree/master/img/Figure_7.png)
