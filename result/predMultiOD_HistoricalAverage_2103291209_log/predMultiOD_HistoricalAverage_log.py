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
from Param_HistoricalAverage import *



def getXSYS(allData, mode):
    TRAIN_NUM = int(allData.shape[0] * trainRatio)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = allData[i:i+TIMESTEP_IN, :, :, :]
            y = allData[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :, :, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  allData.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = allData[i:i+TIMESTEP_IN, :, :, :]
            y = allData[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :, :, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    if MODELNAME == 'CNN':
        XS = XS.transpose(0, 2, 3, 4, 1)
        XS = XS.reshape(XS.shape[0], XS.shape[1], XS.shape[2], -1)
        YS = YS.transpose(0, 2, 3, 4, 1)
        YS = YS.reshape(YS.shape[0], YS.shape[1], YS.shape[2], -1)
    return XS, YS

def HistoricalAverage(XS, YS):
    History = []
    for i in range(0, TIMESTEP_IN, TIMESTEP_OUT):
        History.append(XS[:, i:i+TIMESTEP_OUT, :, :, :])
    History = np.array(History)
#     print(History.shape, np.mean(History, axis=0).shape)
    return np.mean(History, axis=0)

def testModel(name, mode, XS, YS):
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    YS_pred = HistoricalAverage(XS, YS)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('*' * 40)
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())

def trainModel(name, mode, XS, YS):
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    YS_pred = HistoricalAverage(XS, YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('*' * 40)
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())

################# Parameter Setting #######################
MODELNAME = 'HistoricalAverage'
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
    shutil.copy2('Param_HistoricalAverage.py', PATH)
    
    prov_day_data = ss.load_npz(ODPATH)
    prov_day_data_dense = np.array(prov_day_data.todense()).reshape((-1, 47, 47))
    data = prov_day_data_dense[STARTINDEX:ENDINDEX+1,:,:,np.newaxis]
    data = np.log(data + 1.0)
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