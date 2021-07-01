import scipy.sparse as ss
import jpholiday
import sys
import shutil
import os
import time
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import h5py
from copy import copy
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Activation, Flatten, Dense, Reshape, Concatenate, Add, Lambda, Layer, add, multiply, TimeDistributed
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback
import keras.backend as K
import Metrics
from sklearn.preprocessing import StandardScaler
from Param import *
from Param_PCRN import *

def getXSYS(data, mode, dayinfo):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    depends = [[i for i in range(len_c, 0, -1)],
               [i for i in range(len_c + len_p, len_c,  -1)],
               [i for i in range(len_c + len_p + len_t, len_c + len_p,  -1)]]
    if mode == 'TRAIN':
        start = len_c + len_p + len_t
        end = TRAIN_NUM - TIMESTEP_OUT + 1
    elif mode=='TEST':
        start = TRAIN_NUM
        end = data.shape[0] - TIMESTEP_OUT + 1
            
    XC, XP, XT, YS = [], [], [], []
    for i in range(start, end):
        x_c = [data[i - j][np.newaxis, :, :, :] for j in depends[0]]
        x_p = [data[i - j][np.newaxis, :, :, :] for j in depends[1]]
        x_t = [data[i - j][np.newaxis, :, :, :] for j in depends[2]]      
        XC.append(np.vstack(x_c))
        XP.append(np.vstack(x_p))
        XT.append(np.vstack(x_t))
        y = [data[i] for i in range(i,i+TIMESTEP_OUT)]
        YS.append(np.dstack(y))
    XC, XP, XT, YS = np.array(XC), np.array(XP), np.array(XT), np.array(YS)

    if dayinfo:
        DAYS = pd.date_range(start=STARTDATE, end=ENDDATE, freq='1D')
        df = pd.DataFrame()
        df['DAYS'] = DAYS
        df['DAYOFWEEK'] = DAYS.weekday
        df = pd.get_dummies(df, columns=['DAYOFWEEK'])
        df['ISHOLIDAY'] = df['DAYS'].apply(lambda x: int(jpholiday.is_holiday(x) | (x.weekday() >= 5)))
        df = df.drop(columns=['DAYS'])
        day_data = df.values
        YD = []
        for i in range(start, end):
            YD.append(day_data[i:i+TIMESTEP_OUT, :])
        YD = np.array(YD)
        day_info_dim = YD.shape[-1]
    else:
        YD = None
        day_info_dim = 0
        
    XS = [XC, XP, XT, YD] if YD is not None else [XC, XP, XT]
    return XS, YS, day_info_dim

##################### PCRN Model ############################
def ConvLSTMs():
    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         input_shape=(None, HEIGHT, WIDTH, CHANNEL)))
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                         padding='same', return_sequences=False))
    return model


class HadamardFusion(Layer):
    def __init__(self, **kwargs):
        super(HadamardFusion, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.Wc = self.add_weight(name='Wc', shape=(input_shape[0][1:]),
                                  initializer='uniform', trainable=True)
        self.Wp = self.add_weight(name='Wp', shape=(input_shape[1][1:]),
                                  initializer='uniform', trainable=True)
        super(HadamardFusion, self).build(input_shape)

    def call(self, x, mask=None):
        assert isinstance(x, list)
        hct, hallt = x
        hft = K.relu(hct * self.Wc + hallt * self.Wp)
        return hft

    def get_output_shape(self, input_shape):
        return input_shape


def softmax(ej_lst):
    return K.exp(ej_lst[0]) / (K.exp(ej_lst[0]) + K.exp(ej_lst[1]))


def getModel(name, dims):
    TIMESTEP_IN, WIDTH, HEIGHT, CHANNEL, TIMESTEP_OUT, day_info_dim = dims
    x_dim = (TIMESTEP_IN, WIDTH, HEIGHT, CHANNEL)
    if name == 'PCRN':
        # Input xc, xp, xt --> hct1, hP1, hP2
        XC = Input(shape=x_dim)
        XP = Input(shape=x_dim)
        XT = Input(shape=x_dim)

        shared_model = Sequential()
        shared_model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=x_dim))
        shared_model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True))
        shared_model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False))

        hct1 = shared_model(XC)
        hP1 = shared_model(XP)
        hP2 = shared_model(XT)

        # Weighting based fusion
        # daily
        concate1 = Concatenate()([hct1, hP1])
        conv1 = Conv2D(filters=1, kernel_size=(1, 1), padding='same')(concate1)
        flat1 = Flatten()(conv1)
        ej1 = Dense(1)(flat1)

        # weekly
        concate2 = Concatenate()([hct1, hP2])
        conv2 = Conv2D(filters=1, kernel_size=(1, 1), padding='same')(concate2)
        flat2 = Flatten()(conv2)
        ej2 = Dense(1)(flat2)

        aj1 = Lambda(softmax)([ej1, ej2])
        aj2 = Lambda(softmax)([ej2, ej1])
        hPallt = Add()([multiply([aj1, hP1]), multiply([aj2, hP2])])

        hft = HadamardFusion()([hct1, hPallt])

        # transform shape
        hft_reshap = Conv2D(filters=CHANNEL*TIMESTEP_OUT, kernel_size=(3, 3), activation='linear', padding='same')(hft)
        
        # metadata fusion
        if day_info_dim > 0:
            Xmeta = Input(shape=(TIMESTEP_IN, day_info_dim))
            x_day = TimeDistributed(Dense(units=WIDTH * HEIGHT * CHANNEL, activation='linear'))(Xmeta)
            hmeta = Reshape((WIDTH, HEIGHT, CHANNEL*TIMESTEP_OUT))(x_day)
            add2 = Add()([hft_reshap, hmeta])
            X_hat = Activation('linear')(add2)
            model = Model(inputs=[XC, XP, XT, Xmeta], outputs=X_hat)
        else:
            X_hat = hft_reshap
            model = Model(inputs=[XC, XP, XT], outputs=X_hat)
            
        return model
    else:
        return None

def testModel(name, mode, XS, YS, day_info_dim):
    print('Model Evaluation Started ...', time.ctime())
    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    model = getModel(name, (TIMESTEP_IN, WIDTH, HEIGHT, CHANNEL, TIMESTEP_OUT, day_info_dim))
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.load_weights(PATH + '/'+ name + '.h5')
    model.summary()
    keras_score = model.evaluate(XS, YS, verbose=1)
    YS_pred = model.predict(XS, verbose=1, batch_size=BATCHSIZE)
    YS = YS.reshape(YS.shape[0], TIMESTEP_OUT, -1)
    YS = scaler.inverse_transform(YS)
    YS_pred = YS_pred.reshape(YS_pred.shape[0], TIMESTEP_OUT, -1)
    YS_pred = scaler.inverse_transform(YS_pred)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Keras MSE, %.10e, %.10f\n" % (name, mode, keras_score, keras_score))
    f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print("%s, %s, Keras MSE, %.10e, %.10f\n" % (name, mode, keras_score, keras_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())

def trainModel(name, mode, XS, YS, day_info_dim):
    print('Model Training Started ...', time.ctime())
    model = getModel(name, (TIMESTEP_IN, WIDTH, HEIGHT, CHANNEL, TIMESTEP_OUT, day_info_dim))
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()
    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit(XS, YS, batch_size=BATCHSIZE, epochs=EPOCH, shuffle=True,
              callbacks=[csv_logger, checkpointer, LR, early_stopping], validation_split=SPLIT)
    keras_score = model.evaluate(XS, YS, verbose=1)
    YS_pred = model.predict(XS, verbose=1, batch_size=BATCHSIZE)
    YS = YS.reshape(YS.shape[0], TIMESTEP_OUT, -1)
    YS = scaler.inverse_transform(YS)
    YS_pred = YS_pred.reshape(YS_pred.shape[0], TIMESTEP_OUT, -1)
    YS_pred = scaler.inverse_transform(YS_pred)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Keras MSE, %.10e, %.10f\n" % (name, mode, keras_score, keras_score))
    f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print("%s, %s, Keras MSE, %.10e, %.10f\n" % (name, mode, keras_score, keras_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())

################# Parameter Setting #######################
MODELNAME = 'PCRN'
KEYWORD = 'predMultiOD_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M") + '_log'
PATH = FILEPATH + KEYWORD
################# Parameter Setting #######################

###########################Reproducible#############################
import random
np.random.seed(100)
random.seed(100)
os.environ['PYTHONHASHSEED'] = '0'  # necessary for py3
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
tf.set_random_seed(100)
###################################################################

data = ss.load_npz(ODPATH)
data = np.array(data.todense())
data = data[STARTINDEX:ENDINDEX+1, :]
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = data.reshape(-1 ,47, 47, 1)    
print(data.shape, np.min(data), np.max(data))

def main():
    param = sys.argv
    if len(param) == 2:
        GPU = param[-1]
    else:
        GPU = '3'
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = GPU
    set_session(tf.Session(graph=tf.get_default_graph(), config=config))

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_PCRN.py', PATH)

    print('STARTDATE, ENDDATE', STARTDATE, ENDDATE, 'data.shape', data.shape)
    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS, day_info_dim = getXSYS(data, 'TRAIN', DAYINFO)
    print('TRAIN XS.shape YS.shape', [x.shape for x in trainXS], trainYS.shape)
    trainModel(MODELNAME, 'TRAIN', trainXS, trainYS, day_info_dim)

    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS, day_info_dim = getXSYS(data, 'TEST', DAYINFO)
    print('TEST XS.shape YS.shape', [x.shape for x in testXS], testYS.shape)
    testModel(MODELNAME, 'TEST', testXS, testYS, day_info_dim)


if __name__ == '__main__':
    main()