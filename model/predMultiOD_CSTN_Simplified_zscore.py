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
from sklearn.preprocessing import StandardScaler
import Metrics
from Param import *
from CSTN_Simplified import CSTN_Simplified

def getXSYS_single(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :, :, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+1, :, :, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN, data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :, :, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+1, :, :, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    if MODELNAME == 'CSTN_Simplified':
        YS = YS.transpose(0, 2, 3, 4, 1)
        YS = YS.reshape(YS.shape[0], YS.shape[1], YS.shape[2], -1)
    return XS, YS

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
    if name == 'CSTN_Simplified':
        return CSTN_Simplified(TIMESTEP_IN, TIMESTEP_OUT, HEIGHT, WIDTH, CHANNEL)
    else:
        return None

def testModel(name, mode, XS, YS, YS_multi):
    print('Model Evaluation Started ...', time.ctime())
    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    model = getModel(name)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.load_weights(PATH + '/'+ name + '.h5')
    model.summary()
    keras_score = model.evaluate(XS, YS, verbose=1)
    XS_pred_multi, YS_pred_multi = [XS], []
    for i in range(TIMESTEP_OUT):
        XS_tmp = np.concatenate(XS_pred_multi, axis=1)[:, i:, :, :, :]
        YS_pred = model.predict(XS_tmp, verbose=1, batch_size=BATCHSIZE)[:, np.newaxis, :, :, :]
        print('YS_pred.shape, XS_tmp.shape', YS_pred.shape, XS_tmp.shape)
        XS_pred_multi.append(YS_pred)
        YS_pred_multi.append(YS_pred)
    YS_pred_multi = np.concatenate(YS_pred_multi, axis=1)
    YS_multi = YS_multi.reshape(YS_multi.shape[0], TIMESTEP_OUT, -1)
    YS_multi = scaler.inverse_transform(YS_multi)
    YS_pred_multi = YS_pred_multi.reshape(YS_pred_multi.shape[0], TIMESTEP_OUT, -1)
    YS_pred_multi = scaler.inverse_transform(YS_pred_multi)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred_multi)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS_multi)
    print('YS_multi.shape, YS_pred_multi.shape,', YS_multi.shape, YS_pred_multi.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS_multi, YS_pred_multi)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Keras MSE, %.10e, %.10f\n" % (name, mode, keras_score, keras_score))
    f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print("%s, %s, Keras MSE, %.10e, %.10f\n" % (name, mode, keras_score, keras_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Testing Ended ...', time.ctime())

def trainModel(name, mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    model = getModel(name)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()
    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='auto')
    model.fit(XS, YS, batch_size=BATCHSIZE, epochs=EPOCH, shuffle=True,
              callbacks=[csv_logger, checkpointer, LR, early_stopping], validation_split=SPLIT)
    keras_score = model.evaluate(XS, YS, verbose=1)
    YS_pred = model.predict(XS, verbose=1, batch_size=BATCHSIZE)
    YS = YS.reshape(YS.shape[0], -1)
    YS = scaler.inverse_transform(YS)
    YS_pred = YS_pred.reshape(YS_pred.shape[0], -1)
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
MODELNAME = 'CSTN_Simplified'
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
    shutil.copy2('CSTN_Simplified.py', PATH)
    
    print('STARTDATE, ENDDATE', STARTDATE, ENDDATE, 'data.shape', data.shape)
    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS_single(data, 'TRAIN')
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'TRAIN', trainXS, trainYS)
    
    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS_single(data, 'TEST')
    _, testYS_multi = getXSYS(data, 'TEST')
    print('TEST XS.shape, YS.shape, YS_multi.shape', testXS.shape, testYS.shape, testYS_multi.shape)
    testModel(MODELNAME, 'TEST', testXS, testYS, testYS_multi)
    
if __name__ == '__main__':
    main()