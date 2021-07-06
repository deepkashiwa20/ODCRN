import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers import Input, merge, TimeDistributed, Flatten, RepeatVector, Reshape, UpSampling2D, concatenate, add, Dropout, Embedding, Activation
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.engine.topology import Layer
from keras import backend as K

class iLayer(Layer):
    def __init__(self, **kwargs):
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape

def _residual_unit(filters):
    def f(input):
        residual = BatchNormalization()(input)
        residual = Activation('relu')(residual)
        residual = Conv2D(filters, (3, 3), padding='same')(residual)
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(filters, (3, 3), padding='same')(residual)
        return add([input, residual])
    return f

def res_units(filters, repetations):
    def f(input):
        for i in range(repetations):
            input = _residual_unit(filters)(input)
        return input
    return f

def stresnet(c_dim, p_dim, t_dim, y_dim, residual_units, day_info_dim):
    main_inputs, outputs = [], []
    height, width, channel, timestep = y_dim
    output_channel = channel * timestep
    for dim in [c_dim, p_dim, t_dim]:
        map_height, map_width, nb_channel, len_sequence = dim
        input = Input(shape=(map_height, map_width, nb_channel * len_sequence))
        main_inputs.append(input)
        # Conv1
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', data_format='channels_last')(input)
        # Residual Networks
        residual_output = res_units(filters=32, repetations=residual_units)(conv1)
        # Conv2
        activation = Activation('relu')(residual_output)
        conv2 = Conv2D(filters=output_channel, kernel_size=(3, 3), padding='same', data_format='channels_last')(activation)
        outputs.append(conv2)

    # Parametric-matrix-based fusion
    new_outputs = []
    for output in outputs:
        new_outputs.append(iLayer()(output))
    main_output = add(new_outputs)
    
    # Fusing with daytime features
    if day_info_dim > 0:
        day_input = Input(shape=(timestep, day_info_dim))
        main_inputs.append(day_input)
        x_day = TimeDistributed(Dense(units=map_height*map_width*nb_channel, activation='relu'))(day_input)
        day_output = Reshape((map_height, map_width, nb_channel*timestep))(x_day)
        main_output = add([main_output, day_output])
    else:
        print('No Day Information Input.')
    
    main_output = Activation('relu')(main_output)
    model = Model(inputs=main_inputs, outputs=main_output)
    return model

# if __name__ == '__main__':
#     model = stresnet(residual_units=2)
#     model.summary()