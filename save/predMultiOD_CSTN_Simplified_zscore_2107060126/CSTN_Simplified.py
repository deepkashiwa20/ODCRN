from keras.models import load_model, Model, Sequential
from keras.layers import Input, merge, TimeDistributed, Flatten, RepeatVector, Reshape, UpSampling2D, concatenate, add, Dropout, Embedding, Lambda, Concatenate, Dot, Activation, Permute, Add, dot
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras import backend as K
from keras.layers.core import Dense
from Param import *

FILTER = 32

def CSTN_Simplified(timestep_in, timestep_out, height, width, channel):
    od_input = Input(shape=(timestep_in, height, width, channel))
    od_times = [Lambda(lambda x: x[:,i,...])(od_input) for i in range(od_input.shape[1])]
    od_encoded_seq = []
    for step_data in od_times:
        conv1 = Conv2D(filters=FILTER, kernel_size=(3, 3), padding='same')(step_data)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=FILTER, kernel_size=(3, 3), padding='same')(conv1)
        bn2 = BatchNormalization()(conv2)
        lb = Lambda(lambda x: K.expand_dims(x, axis=1))(conv2)
        od_encoded_seq.append(lb)
    od_encoded = Concatenate(axis=1)(od_encoded_seq)
    ft = ConvLSTM2D(filters=FILTER, kernel_size=(3,3), padding='same', return_sequences=False)(od_encoded)
    ft = Conv2D(filters=FILTER, kernel_size=(1, 1), padding='same', activation='relu')(ft)
    
    fusion0 = Conv2D(filters=FILTER, kernel_size=(1, 1), padding='same', activation='relu')(ft)
    fusion0 = Reshape((-1, FILTER))(fusion0)
    fusion0 = Permute((2, 1))(fusion0)
    fusion1 = Conv2D(filters=FILTER, kernel_size=(1, 1), padding='same', activation='relu')(ft)
    fusion1 = Reshape((-1, FILTER))(fusion1)
    fusion1 = Permute((2, 1))(fusion1)
    fusion2 = Conv2D(filters=FILTER, kernel_size=(1, 1), padding='same', activation='relu')(ft)
    fusion2 = Reshape((-1, FILTER))(fusion2)
    fusion2 = Permute((2, 1))(fusion2)
    sim_fusion = Dot(axes=(1,1), normalize=True)([fusion1, fusion2])
    softmax = Activation("softmax")(sim_fusion)
    softmaxfusion = Dot(axes=[2,2])([softmax, fusion0])
    softmaxfusion = Reshape((HEIGHT, WIDTH, -1))(softmaxfusion)
    fusion = Add()([ft, softmaxfusion])
    
    softmaxfusion = Reshape((HEIGHT, WIDTH, -1))(fusion)
    output = Conv2D(1, kernel_size=(1, 1), padding='same', activation='linear')(softmaxfusion)
    model = Model(inputs=od_input, outputs=output)
    return model