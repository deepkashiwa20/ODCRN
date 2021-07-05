import sys
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from datetime import datetime
import time
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.preprocessing import normalize
import random
from sklearn.preprocessing import StandardScaler
import Metrics
from Param import *
from GEML import *
from Param_GEML import *
import jpholiday


def getXSYS_single(data, mode):
    DAYS = pd.date_range(start=STARTDATE, end=ENDDATE, freq='1D')
    df = pd.DataFrame()
    df['DAYS'] = DAYS
    df['DAYOFWEEK'] = DAYS.weekday
    df = pd.get_dummies(df, columns=['DAYOFWEEK'])
    df['ISHOLIDAY'] = df['DAYS'].apply(lambda x: int(jpholiday.is_holiday(x) | (x.weekday() >= 5)))
    df = df.drop(columns=['DAYS'])
    YD = df.values

    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + 1, :, :]
            XS.append(x), YS.append(y)
        YD = YD[TIMESTEP_IN:TRAIN_NUM - TIMESTEP_OUT + 1]
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN, data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + 1, :, :]
            XS.append(x), YS.append(y)
        YD = YD[TRAIN_NUM:]
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS, YD


def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN, data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS


def get_adj(data):
    W_adj = np.load(ADJPATH)
    print('W_adj = np.load(ADJPATH)###########################', W_adj.shape)
    W_adj = normalize(W_adj, norm='l1')
    W_adj[range(W_adj.shape[0]), range(W_adj.shape[1])] += 1
    W_adj = np.array([W_adj for i in range(TIMESTEP_IN)])
    W_adj = np.array([W_adj for i in range(data.shape[0])])
    print('data.shape', data.shape)
    W_seman = (data + data.transpose((0, 1, 3, 2)))
    for i in range(W_seman.shape[0]):
        for j in range(W_seman.shape[1]):
            W_seman[i, j] = normalize(W_seman[i, j], norm='l1')
            W_seman[i, j][range(W_seman.shape[2]), range(W_seman.shape[3])] += 1
    return W_adj, W_seman


def trainModel(name, mode, XS, YS, YD):
    print('Model Training Started ...', time.ctime())
    model = getModel()
    W_adj, W_seman = get_adj(XS)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()
    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='auto')
    model.fit(x=[XS, W_adj, W_seman, YD], y=YS, batch_size=BATCHSIZE, epochs=EPOCH, verbose=1, 
              validation_split=SPLIT, callbacks=[csv_logger, checkpointer, LR, early_stopping])
    keras_score = model.evaluate(x=[XS, W_adj, W_seman, YD], y=YS, batch_size=1)
    YS_pred = model.predict(x=[XS, W_adj, W_seman, YD], batch_size=1)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
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

def testModel(name, mode, XS, YS, YS_multi, YD):
    print('Model Testing Started ...', time.ctime())
    W_adj, W_seman = get_adj(XS)
    model = getModel()
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.load_weights(PATH + '/' + name + '.h5')
    model.summary()
    keras_score = model.evaluate(x=[XS, W_adj, W_seman, YD[:XS.shape[0]]], y=YS, batch_size=1)
    XS_pred_multi, YS_pred_multi = [XS], []
    for i in range(TIMESTEP_OUT):
        tmp_torch = np.concatenate(XS_pred_multi, axis=1)[:, i:, :, :]
        _, W_seman = get_adj(tmp_torch)
        YS_pred = model.predict(x=[tmp_torch, W_adj, W_seman, YD[i:XS.shape[0] + i]], batch_size=1)
        print('type(YS_pred), YS_pred.shape, XS_tmp_torch.shape', type(YS_pred), YS_pred.shape, tmp_torch.shape)
        XS_pred_multi.append(YS_pred)
        YS_pred_multi.append(YS_pred)
    YS_pred_multi = np.concatenate(YS_pred_multi, axis=1)
    print('YS_multi.shape, YS_pred_multi.shape,', YS_multi.shape, YS_pred_multi.shape)
    YS_multi = YS_multi.reshape(YS_multi.shape[0], TIMESTEP_OUT, -1)
    YS_multi = scaler.inverse_transform(YS_multi)
    YS_pred_multi = YS_pred_multi.reshape(YS_pred_multi.shape[0], TIMESTEP_OUT, -1)
    YS_pred_multi = scaler.inverse_transform(YS_pred_multi)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred_multi)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS_multi)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS_multi, YS_pred_multi)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Keras MSE, %.10e, %.10f\n" % (name, mode, keras_score, keras_score))
    f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print("%s, %s, Keras MSE, %.10e, %.10f\n" % (name, mode, keras_score, keras_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Testing Ended ...', time.ctime())


################# Parameter Setting #######################
MODELNAME = 'GEML'
KEYWORD = 'predMultiOD_' + MODELNAME + '_zscore_' + datetime.now().strftime("%y%m%d%H%M")
PATH = FILEPATH + KEYWORD
BATCHSIZE = 1
os.environ['PYTHONHASHSEED'] = '0'
tf.set_random_seed(100)
np.random.seed(100)
random.seed(100)
###########################################################
param = sys.argv
if len(param) == 2:
    GPU = param[-1]
else:
    GPU = '3'
config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = GPU
set_session(tf.Session(graph=tf.get_default_graph(), config=config))
###########################################################

data = ss.load_npz(ODPATH)
data = np.array(data.todense())
data = data[STARTINDEX:ENDINDEX+1, :]
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = data.reshape(-1, 47, 47)    
print(data.shape, np.min(data), np.max(data))

def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_GEML.py', PATH)
    shutil.copy2('GEML.py', PATH)
    
    print('STARTDATE, ENDDATE', STARTDATE, ENDDATE, 'data.shape', data.shape)
    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS, YD = getXSYS_single(data, 'TRAIN')
    print('TRAIN XS.shape YD.shape YS,shape', trainXS.shape, YD.shape, trainYS.shape)
    trainModel(MODELNAME, 'TRAIN', trainXS, trainYS, YD)

    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS, YD = getXSYS_single(data, 'TEST')
    _, testYS_multi = getXSYS(data, 'TEST')
    print('TEST XS.shape, YD.shape, YS.shape, YS_multi.shape', testXS.shape, YD.shape, testYS.shape, testYS_multi.shape)
    testModel(MODELNAME, 'TEST', testXS, testYS, testYS_multi, YD)


if __name__ == '__main__':
    main()
