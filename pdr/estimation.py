##########################################################################################################
#                                                                                                        #
#             Main function for 6-DOF estimation:                                                        #
#                - train the neural network with different parameters selected by the user               #
#                - predict the world coordinates position on testing IMU data                            #
#                - display ground truth and estimation for each prediction in real time                  #
#                                                                                                        #
##########################################################################################################
import os
import tensorflow as tf
from datetime import datetime
from sklearn.utils import shuffle
from pdr.utils.data.data_loader import data_loader
from pdr.utils.data.data_processing import augmented_preprocessing, preprocessing, augmented_predictions_processing, predictions_processing
from pdr.utils.models.build_model import build_model, build_multi_loss_model, multi_loss_layer
from pdr.utils.plot import real_time_3d_plotting
from pdr.utils.data.data_augmentation import linear_interpolation
from pdr.utils.geometry.world_coordinates import world_coordinates_from_rot_trans


def run(arcore_paths, imu_paths, test_arcore_paths, test_imu_paths):
    try:

        ############################
        # Load and preprocess data #
        ############################
        arcore = data_loader(arcore_paths, 'arcore')
        imu = data_loader(imu_paths, 'imu')
        os.system('clear')
        while True:
            try:
                augm = input('Do you want to use data augmentation ?(y/[n])\n')
                lookback = int(input('Please enter lookback size:\n'))
                step = int(input('Please enter step length:\n'))
                if augm == 'y':
                    augm = True
                    break
                elif not augm or augm == 'n':
                    augm = False
                    break
                else:
                    raise ValueError
            except ValueError:
                print('Oops wrong input !')

        if augm:
            augmented_data = linear_interpolation(arcore, imu)
            (acc_samples, gyr_samples), (t, q) = augmented_preprocessing(
                augmented_data, lookback, step)
        else:
            (acc_samples, gyr_samples), (t, q) = preprocessing(
                arcore, imu, lookback, step)
        acc_samples, gyr_samples, t, q = shuffle(
            acc_samples, gyr_samples, t, q)
        #####################################################################################################
        # Define Keras callbacks to save the best model and follow train and validation loss on Tensorboard #
        #####################################################################################################
        date = str(datetime.now().strftime("%Y-%m-%d_%H:%M"))
        base_path = os.path.abspath('pdr')
        callbacks = [
            # tf.keras.callbacks.TensorBoard(
            #     log_dir=base_path + "/data/logs/"+date
            # ),
            tf.keras.callbacks.ModelCheckpoint(
                base_path + '/data/models/'+date+'_6dof_model.hdf5', monitor='val_loss', save_best_only=True, verbose=1, save_weights_only=True)
        ]

        ################################################################################
        # Build CNN LSTM model using multiloss model to optimize loss function weights #
        # and train it according to user batch_size and epochs selection               #
        ################################################################################
        tf.keras.backend.clear_session()
        model = build_model(lookback)
        loss_model = build_multi_loss_model(model, lookback)
        loss_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=None)
        os.system('clear')
        while True:
            try:
                batch_size = int(input('Please enter batch size:\n'))
                epochs = int(input('Please enter epochs number:\n'))
                break
            except ValueError:
                print("Oops ! That's not a valid number")
        loss_model.fit([acc_samples, gyr_samples, t, q], epochs=epochs,
                       batch_size=batch_size, callbacks=callbacks, validation_split=0.1, verbose=1)
        os.system('clear')

        ######################################################################################################################
        # Load the best model saved during the training session and create a new prediction model without the multiloss part #
        # with trained model weigths                                                                                          #
        ######################################################################################################################
        print('Loading best model')
        tf.keras.backend.clear_session()
        prediction_model = build_model(lookback)
        trained_model = build_multi_loss_model(prediction_model, lookback)
        trained_model.load_weights(
            base_path + '/data/models/'+date+'_6dof_model.hdf5')
        prediction_model.set_weights(trained_model.get_weights()[:-2])

        ######################################
        # Load and pre-process testing data  #
        ######################################
        test_arcore = data_loader(test_arcore_paths, 'arcore')
        test_imu = data_loader(test_imu_paths, 'imu')

        if augm:
            augmented_data = linear_interpolation(test_arcore, test_imu)
            init_coord, init_q, (acc_test, gyr_test), (t_test, q_test) = augmented_predictions_processing(
                augmented_data, lookback, step)
        else:
            init_coord, init_q, (acc_test, gyr_test), (t_test, q_test) = predictions_processing(
                test_arcore, test_imu, lookback, step)
        os.system('clear')

        while True:
            try:
                res = int(input(
                    "To predict testing data and see the trajectories in real time, type '0'\nTo convert the model in a tflite format, type '1'"))
                if res == 0:
                    ###########################
                    # Predict and plot result #
                    ###########################
                    print('Predictions')
                    for i in range(len(test_arcore)):
                        prediction = prediction_model.predict(
                            {'acc': acc_test[i], 'gyr': gyr_test[i]}, verbose=1, batch_size=1)
                        (x, y, z) = world_coordinates_from_rot_trans(
                            init_coord[i], init_q[i], t_test[i], q_test[i])
                        (x_estimated, y_estimated, z_estimated) = world_coordinates_from_rot_trans(
                            init_coord[i], init_q[i], prediction[0], prediction[1])
                        real_time_3d_plotting(
                            0.5, (x, y, z), (x_estimated, y_estimated, z_estimated))
                    break
                if res == 1:
                    prediction_model.save(
                        base_path + '/data/models/'+date+'_model.hdf5')
                    print('Convert model to tflite format')
                    converter = tf.lite.TFLiteConverter.from_keras_model_file(
                        base_path + '/data/models/'+date+'_model.hdf5')
                    tflite_model = converter.convert()
                    open(base_path + '/data/models/'+date +
                         '.tflite', "wb").write(tflite_model)
                    break
                else:
                    raise ValueError
            except ValueError:
                print("Oops ! That's not a valid number")

    except Exception as e:
        print("Exeption {} occured, arguments:\n{!r}".format(
            type(e).__name__, e.args))
    return
