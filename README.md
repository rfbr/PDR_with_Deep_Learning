# Deep learning based pedestrian dead reckoning using smartphone IMU

This project was part of my research internship at Kyushu University in Japan, supervised by professor Hideaki Uchiyama. 
It is based on above works: 

https://github.com/jpsml/6-DOF-Inertial-Odometry
https://www.mdpi.com/1424-8220/19/17/3777/pdf

Introduction
=
Selfsufficient running and independance from the outside environment give smarthpone built-in inertial sensors an important role in indoor navigation. This repository allows the creation of a mobile real-time indoor positioning system using a smart-phone held in hand. 
In order to achieve this, we will only use the Inertial Measurement Unit (IMU) present in the phone, gathering acceleration and angular velocity. The proposed method is based on 6-DOF odometry, computing relative sensor pose changes between two sequential moments using a neural network. The estimate will then be refined by a Bayesian filter, a particle filter, associated with a map-matching algorithm.

 This repository contains only the 6-DOF odometry implementation using IMU data gathered with an android application (you can find the source code [here](https://github.com/rfbr/IMU_and_pose_Android_Recorder)), powered by a CNN-LSTM neural network. It allows to create a tflite model with the best trained model, ready to be used in our android application.

Global trajectory estimation
=
To estimate the pedestrian 3D world coordinates, we use our neural network to predict 6DOF relative coordinates (represented by a translation vector and a quaternion rotation vector) between two given timestamps separated by a constant step.
Thus, by combining all relative poses, we can compute the global pose.

Data format
=
We use IMU data (acceleration and angular velocity) gathered by an android application via embedded accelerometer and gyroscope to estimate the ground truth position captured via ARCore.

The neural network
=
We tryed different architectures and kept the best one according to the validation loss (tracked with TensorBoard):
![lolz](https://user-images.githubusercontent.com/45492759/69095785-5d2cc580-0a53-11ea-8c4d-15a0c374ebb5.png)
The architecture corresponding is the following:
![NN(1)](https://user-images.githubusercontent.com/45492759/69096096-0ecbf680-0a54-11ea-820e-9668266872a6.png)

Our neural network can be divided into three main parts:
- A convolutional part: accelerometer and gyroscope data are preprocessed separatly. We use 3 layers to extract local features (1D convolutional layer/max pooling/1D convolutional layer) and then concatenate the results.
- A LSTM part: we use two LSTM layers to process preprocessed IMU data as a time series.
- A multi-loss layer: this part is used to optimize the different weights between the different loss functions in use, according to the multi-task learning theory [[1]](https://arxiv.org/abs/1705.07115). 

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
![1](https://user-images.githubusercontent.com/45492759/64983234-5ac9b600-d8c0-11e9-87c0-d53ada211f5f.png)
![2](https://user-images.githubusercontent.com/45492759/64983238-5bfae300-d8c0-11e9-9af7-31400fd39063.png)
![4](https://user-images.githubusercontent.com/45492759/64983246-5f8e6a00-d8c0-11e9-92b1-91ffb40a1db4.png)
- With augmented training data (linear interpolation):
![Figure_1](https://user-images.githubusercontent.com/45492759/64983597-2dc9d300-d8c1-11e9-814e-1b0e227e9efe.png)
![Figure_2](https://user-images.githubusercontent.com/45492759/64983598-2e626980-d8c1-11e9-8436-0b445d134c62.png)
![Figure_4](https://user-images.githubusercontent.com/45492759/64983603-2f939680-d8c1-11e9-8088-138280f54684.png)


Concerning floor changing detection, data augmentation leads to bad results. Spherical linear interpolation for quaternions might not be the best or we may have done some mistakes: 
- With normal training data:
![6](https://user-images.githubusercontent.com/45492759/64983633-3f12df80-d8c1-11e9-88ee-43b68b174306.png)
![7](https://user-images.githubusercontent.com/45492759/64983637-3fab7600-d8c1-11e9-9296-0be589ef1a5d.png)
- With augmented training data (linear interpolation):
![Figure_6](https://user-images.githubusercontent.com/45492759/64983644-41753980-d8c1-11e9-96ff-b014a8895275.png)
![Figure_7](https://user-images.githubusercontent.com/45492759/64983797-a6c92a80-d8c1-11e9-9411-5535b6f9ef9b.png)
