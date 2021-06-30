import pandas as pd
import scipy.sparse as ss
import csv
import numpy as np
import os
import shutil
import sys
import time
from datetime import datetime
import Metrics
from Param import *
from Param_HistoricalAverage import *
from sklearn.preprocessing import StandardScaler

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
    return np.mean(History, axis=0)

def testModel(name, mode, XS, YS):
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    YS_pred = HistoricalAverage(XS, YS)
    YS = YS.reshape(YS.shape[0], TIMESTEP_OUT, -1)
    YS = scaler.inverse_transform(YS)
    YS_pred = YS_pred.reshape(YS_pred.shape[0], TIMESTEP_OUT, -1)
    YS_pred = scaler.inverse_transform(YS_pred)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Testing Ended ...', time.ctime())

def trainModel(name, mode, XS, YS):
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    YS_pred = HistoricalAverage(XS, YS)
    YS = YS.reshape(YS.shape[0], TIMESTEP_OUT, -1)
    YS = scaler.inverse_transform(YS)
    YS_pred = YS_pred.reshape(YS_pred.shape[0], TIMESTEP_OUT, -1)
    YS_pred = scaler.inverse_transform(YS_pred)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())

################# Parameter Setting #######################
MODELNAME = 'HistoricalAverage'
KEYWORD = 'predMultiOD_' + MODELNAME + '_zscore_' + datetime.now().strftime("%y%m%d%H%M")
PATH = FILEPATH + KEYWORD
################# Parameter Setting #######################

data = ss.load_npz(ODPATH)
data = np.array(data.todense())
data = data[STARTINDEX:ENDINDEX+1, :]
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = data.reshape(-1 ,47, 47, 1)    
print(data.shape, np.min(data), np.max(data))

def main():    
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