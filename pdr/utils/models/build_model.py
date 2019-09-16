##########################################################################################################
#                                                                                                        #
#             Define neural network architecture and the multiloss layer for 6DOF estimation             #
#                                                                                                        #
##########################################################################################################
import tensorflow as tf
import tfquaternion as tfq

########################
# Build CNN+LSTM model #
########################

NUMBER_OF_NEURONS_CONV = 96
NUMBER_OF_NEURONS_LSTM = 96
KERNEL_SIZE = 15


def build_model(lookback):

    # Inputs
    acc_inputs = tf.keras.Input(shape=(lookback, 3), name='acc')
    gyr_inputs = tf.keras.Input(shape=(lookback, 3), name='gyr')

    # Acc branch
    x = tf.keras.layers.Conv1D(
        NUMBER_OF_NEURONS_CONV, KERNEL_SIZE, padding='causal', name='acc_conv_1')(acc_inputs)
    x = tf.keras.layers.BatchNormalization(name='acc_batch_1')(x)
    x = tf.keras.layers.ReLU(name='acc_relu_1')(x)
    x = tf.keras.layers.AveragePooling1D(3, name='acc_av_1')(x)
    x = tf.keras.layers.Conv1D(
        NUMBER_OF_NEURONS_CONV, KERNEL_SIZE, activation='relu', padding='causal', name='acc_conv_2')(x)
    x = tf.keras.layers.BatchNormalization(name='acc_batch_2')(x)
    x = tf.keras.layers.ReLU(name='acc_relu_2')(x)

    x = tf.keras.layers.Conv1D(
        NUMBER_OF_NEURONS_CONV, KERNEL_SIZE, padding='causal', name='acc_conv_3')(x)
    x = tf.keras.layers.BatchNormalization(name='acc_batch_3')(x)
    x = tf.keras.layers.ReLU(name='acc_relu_3')(x)
    x = tf.keras.layers.AveragePooling1D(3, name='acc_av_2')(x)
    x = tf.keras.layers.Conv1D(
        NUMBER_OF_NEURONS_CONV, KERNEL_SIZE, activation='relu', padding='causal', name='acc_conv_4')(x)
    x = tf.keras.layers.BatchNormalization(name='acc_batch_4')(x)
    x = tf.keras.layers.ReLU(name='acc_relu_4')(x)

    # Gyr branch
    y = tf.keras.layers.Conv1D(
        NUMBER_OF_NEURONS_CONV, KERNEL_SIZE, padding='causal', name='gyr_conv_1')(gyr_inputs)
    y = tf.keras.layers.BatchNormalization(name='gyr_batch_1')(y)
    y = tf.keras.layers.ReLU(name='gyr_relu_1')(y)
    y = tf.keras.layers.AveragePooling1D(3, name='gyr_av_1')(y)
    y = tf.keras.layers.Conv1D(
        NUMBER_OF_NEURONS_CONV, KERNEL_SIZE, activation='relu', padding='causal', name='gyr_conv_2')(y)
    y = tf.keras.layers.BatchNormalization(name='gyr_batch_2')(y)
    y = tf.keras.layers.ReLU(name='gyr_relu_2')(y)

    y = tf.keras.layers.Conv1D(
        NUMBER_OF_NEURONS_CONV, KERNEL_SIZE, padding='causal', name='gyr_conv_3')(y)
    y = tf.keras.layers.BatchNormalization(name='gyr_batch_3')(y)
    y = tf.keras.layers.ReLU(name='gyr_relu_3')(y)
    y = tf.keras.layers.AveragePooling1D(3, name='gyr_av_2')(y)
    y = tf.keras.layers.Conv1D(
        NUMBER_OF_NEURONS_CONV, KERNEL_SIZE, activation='relu', padding='causal', name='gyr_conv_4')(y)
    y = tf.keras.layers.BatchNormalization(name='gyr_batch_4')(y)
    y = tf.keras.layers.ReLU(name='gyr_relu_4')(y)

    # Concatenate
    concatenate = tf.keras.layers.concatenate(
        [x, y], axis=1, name='concatenate')

    # LSTM
    z = tf.keras.layers.Lambda(buildLstmLayer, arguments={
                               'num_layers': 2, 'num_units': NUMBER_OF_NEURONS_LSTM}, name='lstm')(concatenate)
    z = tf.keras.layers.Dropout(0.25, name='dropout')(z)
    translation_prediction = tf.keras.layers.Dense(3, name='translation')(z)
    rotation_prediction = tf.keras.layers.Dense(4, name='rotation')(z)
    model = tf.keras.Model([acc_inputs, gyr_inputs], [
                           translation_prediction, rotation_prediction])

    return model


def buildLstmLayer(inputs, num_layers, num_units):
    with tf.compat.v1.variable_scope("foo", reuse=tf.compat.v1.AUTO_REUSE):
        lstm_cells = []
        for i in range(num_layers):
            lstm_cells.append(
                tf.compat.v1.lite.experimental.nn.TFLiteLSTMCell(
                    num_units, forget_bias=0, name='rnn{}'.format(i)))
        lstm_layers = tf.keras.layers.StackedRNNCells(lstm_cells)
        # Assume the input is sized as [batch, time, input_size], then we're going
        # to transpose to be time-majored.
        transposed_inputs = tf.transpose(
            inputs, perm=[1, 0, 2])
        outputs, _ = tf.compat.v1.lite.experimental.nn.dynamic_rnn(
            lstm_layers,
            transposed_inputs,
            dtype='float32',
            time_major=True)
        unstacked_outputs = tf.unstack(outputs, axis=0)
    return unstacked_outputs[-1]

###################################
# Define quaternion loss function #
###################################


def quat_mult_error(y_true, y_pred):
    q_hat = tfq.Quaternion(y_true)
    q = tfq.Quaternion(y_pred).normalized()
    q_prod = q * q_hat.conjugate()
    _, x, y, z = tf.split(q_prod, num_or_size_splits=4, axis=-1)
    return tf.abs(tf.multiply(2.0, tf.concat(values=[x, y, z], axis=-1)))


def quaternion_mean_multiplicative_error(y_true, y_pred):
    return tf.reduce_mean(quat_mult_error(y_true, y_pred))

###########################
# Define multi loss model #
###########################


class multi_loss_layer(tf.keras.layers.Layer):

    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(multi_loss_layer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var'+str(i), shape=(
                1,), initializer=tf.keras.initializers.Constant(0.), trainable=True)]
        super(multi_loss_layer, self).build(input_shape)

    def multi_loss(self, y_true, y_pred):
        assert len(y_true) == self.nb_outputs and len(
            y_pred) == self.nb_outputs
        loss = 0
        precision = tf.keras.backend.exp(-self.log_vars[0][0])
        loss += precision * \
            tf.keras.losses.mean_absolute_error(
                y_true[0], y_pred[0]) + self.log_vars[0][0]
        precision = tf.keras.backend.exp(-self.log_vars[1][0])
        loss += precision * \
            quaternion_mean_multiplicative_error(
                y_true[1], y_pred[1]) + self.log_vars[1][0]
        return tf.keras.backend.mean(loss)

    def get_config(self):
        base_config = super(multi_loss_layer, self).get_config()
        return base_config

    def call(self, inputs):
        y_true = inputs[:self.nb_outputs]
        y_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(y_true, y_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return tf.keras.backend.concatenate(inputs, -1)

######################################################################################
# Compile both models to train the neural network and also the loss function weigths #
######################################################################################


def build_multi_loss_model(model, lookback):
    acc_inputs = tf.keras.layers.Input(
        shape=(lookback, 3), name='multi_loss_acc')
    gyr_inputs = tf.keras.layers.Input(
        shape=(lookback, 3), name='multi_loss_gyr')
    t, q = model([acc_inputs, gyr_inputs])
    t_true = tf.keras.layers.Input(shape=(3,), name='t_true')
    q_true = tf.keras.layers.Input(shape=(4,), name='q_true')
    outputs = multi_loss_layer(nb_outputs=2)([t_true, q_true, t, q])
    multi_loss_model = tf.keras.models.Model(
        [acc_inputs, gyr_inputs, t_true, q_true], outputs)
    return multi_loss_model
