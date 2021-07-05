import scipy.sparse as ss
import jpholiday
import csv
import numpy as np
import os
import shutil
import sys
import time
import pandas as pd
from datetime import datetime
from keras.models import load_model, Model, Sequential
from keras.layers import Input, merge, TimeDistributed, Flatten, RepeatVector, Reshape, UpSampling2D, concatenate, add, Dropout, Embedding
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler
import Metrics
from sklearn.preprocessing import StandardScaler
from Param import *
from Param_STResNet import *
from STResNet import stresnet

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
        x_c = [data[i - j] for j in depends[0]]
        x_p = [data[i - j] for j in depends[1]]
        x_t = [data[i - j] for j in depends[2]]
        y = [data[i] for i in range(i,i+TIMESTEP_OUT)]
        XC.append(np.dstack(x_c))
        XP.append(np.dstack(x_p))
        XT.append(np.dstack(x_t))
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


def getModel(name, nb_res_units, day_info_dim):
    if name == 'STResNet':
        c_dim = (HEIGHT, WIDTH, nb_channel, len_c)
        p_dim = (HEIGHT, WIDTH, nb_channel, len_p)
        t_dim = (HEIGHT, WIDTH, nb_channel, len_t)
        y_dim = (HEIGHT, WIDTH, nb_channel, TIMESTEP_OUT)
        model = stresnet(c_dim = c_dim, p_dim = p_dim, t_dim = t_dim, y_dim = y_dim,
                         residual_units = nb_res_units, filters=nb_filters, day_info_dim = day_info_dim)
        return model
    else:
        return None
    
def testModel(name, mode, XS, YS, day_info_dim):
    print('Model Evaluation Started ...', time.ctime())
    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    model = getModel(name, nb_res_units, day_info_dim)
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
    print('Model Evaluation Ended ...', time.ctime())

def trainModel(name, mode, XS, YS, day_info_dim):
    print('Model Training Started ...', time.ctime())
    model = getModel(name, nb_res_units, day_info_dim)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()
    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='auto')
    model.fit(XS, YS, batch_size=BATCHSIZE, epochs=EPOCH, shuffle=False, callbacks=[csv_logger, checkpointer, LR, early_stopping], validation_split=SPLIT)
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
MODELNAME = 'STResNet'
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
    shutil.copy2('Param_STResNet.py', PATH)
    shutil.copy2('STResNet.py', PATH)

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