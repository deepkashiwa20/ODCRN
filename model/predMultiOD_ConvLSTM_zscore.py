import pandas as pd
import scipy.sparse as ss
import csv
import numpy as np
import os
import shutil
import sys
import time
from datetime import datetime
from keras import backend as K
from keras.models import load_model, Model, Sequential
from keras.layers import Input, merge, TimeDistributed, Flatten, RepeatVector, Reshape, UpSampling2D, concatenate, add, Dropout, Embedding, Lambda
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler
import Metrics
from Param import *
from sklearn.preprocessing import StandardScaler

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :, :, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :, :, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :, :, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :, :, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS
    
def getModel(name):
    if name == 'ConvLSTM':
        input = Input(shape=(TIMESTEP_IN, HEIGHT, WIDTH, CHANNEL))
        x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(input)
        x = BatchNormalization()(x)
        x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Lambda(lambda x: K.concatenate([x[:, np.newaxis, :, :, :]] * TIMESTEP_OUT, axis=1))(x)
        x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = ConvLSTM2D(filters=CHANNEL, kernel_size=(3, 3), padding='same', return_sequences=True, activation='linear')(x)
        model = Model(inputs=[input], outputs=[x])
        return model
    else:
        return None

def testModel(name, mode, XS, YS):
    print('BATCHSIZE, LOSS, LEARN, OPTIMIZER', BATCHSIZE, LOSS, LEARN, OPTIMIZER)
    print('Model Evaluation Started ...', time.ctime())
    model = getModel(name)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
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
    print('*' * 40)
    print("%s, %s, Keras MSE, %.10e, %.10f\n" % (name, mode, keras_score, keras_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Evaluation Ended ...', time.ctime())

def trainModel(name, mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    model = getModel(name)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()
    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='auto')
    model.fit(XS, YS, batch_size=BATCHSIZE, epochs=EPOCH, callbacks=[csv_logger, checkpointer, LR, early_stopping], validation_split=SPLIT)
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
    print('*' * 40)
    print("%s, %s, Keras MSE, %.10e, %.10f\n" % (name, mode, keras_score, keras_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())

################# Parameter Setting #######################
MODELNAME = 'ConvLSTM'
KEYWORD = 'predMultiOD_' + MODELNAME + '_zscore_' + datetime.now().strftime("%y%m%d%H%M")
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
    
    print('STARTDATE, ENDDATE', STARTDATE, ENDDATE, 'data.shape', data.shape)
    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'TRAIN', trainXS, trainYS)

    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS(data, 'TEST')
    print('TEST XS.shape YS,shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'TEST', testXS, testYS)


if __name__ == '__main__':
    main()