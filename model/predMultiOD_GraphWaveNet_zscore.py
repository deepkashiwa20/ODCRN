import sys
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
from sklearn.preprocessing import StandardScaler
from GraphWaveNet import *
import Metrics
from Param import *
    
def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :, :, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :, :, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN, data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :, :, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :, :, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    if MODELNAME == 'GraphWaveNet':
        XS = XS.transpose(0, 2, 3, 4, 1)
        XS = XS.reshape(XS.shape[0], XS.shape[1], XS.shape[2], -1)
        YS = YS.transpose(0, 1, 3, 2, 4)
        YS = YS.reshape(YS.shape[0], -1, YS.shape[3], YS.shape[4])
    return XS, YS

def getModel(name):
    if name == 'GraphWaveNet':
        W = np.load(ADJPATH).astype(np.float32)
        adj_mx = load_adj(W)
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        model = gwnet(device, in_dim = 47, out_dim = 47*TIMESTEP_OUT, supports=supports).to(device)
        summary(model, (47, 47, TIMESTEP_IN), device=device)
        return model
    else:
        return None
    
def evaluate_model(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n
    
def trainModel(name, mode, XS, YS):
    print('Model Training Started ...', time.ctime())    
    model = getModel(name)

    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * 0.8)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE)
    
    if LOSS == 'mse':
        criterion = nn.MSELoss()
    if LOSS == 'mae':
        criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)

    min_val_loss = np.inf
    wait = 0
    for epoch in range(EPOCH):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n       
        
        val_loss = evaluate_model(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + name + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, ", validation loss:", val_loss)
    
    torch_score = evaluate_model(model, criterion, train_iter)
    model.eval()
    with torch.no_grad():
        YS_pred = model(XS_torch).cpu().numpy()
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS = YS.reshape(YS.shape[0], -1, 47, 47)
    YS = YS.transpose(0, 1, 3, 2)
    YS = YS.reshape(YS.shape[0], TIMESTEP_OUT, -1)
    YS = scaler.inverse_transform(YS)
    YS_pred = YS_pred.reshape(YS_pred.shape[0], -1, 47, 47)
    YS_pred = YS_pred.transpose(0, 1, 3, 2)
    YS_pred = YS_pred.reshape(YS_pred.shape[0], TIMESTEP_OUT, -1)
    YS_pred = scaler.inverse_transform(YS_pred)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())
        
def testModel(name, mode, XS, YS):
    print('Model Testing Started ...', time.ctime())
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    
    if LOSS == 'mse':
        criterion = nn.MSELoss()
    if LOSS == 'mae':
        criterion = nn.L1Loss()
        
    torch_score = evaluate_model(model, criterion, test_iter)
    model.eval()
    with torch.no_grad():
        YS_pred = model(XS_torch).cpu().numpy()
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS = YS.reshape(YS.shape[0], -1, 47, 47)
    YS = YS.transpose(0, 1, 3, 2)
    YS = YS.reshape(YS.shape[0], TIMESTEP_OUT, -1)
    YS = scaler.inverse_transform(YS)
    YS_pred = YS_pred.reshape(YS_pred.shape[0], -1, 47, 47)
    YS_pred = YS_pred.transpose(0, 1, 3, 2)
    YS_pred = YS_pred.reshape(YS_pred.shape[0], TIMESTEP_OUT, -1)
    YS_pred = scaler.inverse_transform(YS_pred)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    print('YS_multi.shape, YS_pred_multi.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Testing Ended ...', time.ctime())
        
################# Parameter Setting #######################
MODELNAME = 'GraphWaveNet'
KEYWORD = 'predMultiOD_' + MODELNAME + '_zscore_' + datetime.now().strftime("%y%m%d%H%M")
PATH = FILEPATH + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
torch.backends.cudnn.deterministic = True
###########################################################
param = sys.argv
if len(param) == 2:
    GPU = param[-1]
else:
    GPU = '3'
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
###########################################################

data = ss.load_npz(ODPATH)
data = np.array(data.todense())
data = data[STARTINDEX:ENDINDEX+1, :]
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = data.reshape(-1, 47, 47, 1)    
print(data.shape, np.min(data), np.max(data))

def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('GraphWaveNet.py', PATH)
        
    print('STARTDATE, ENDDATE', STARTDATE, ENDDATE, 'data.shape', data.shape)
    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'TRAIN', trainXS, trainYS)
    
    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS(data, 'TEST')
    print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'TEST', testXS, testYS)

    
if __name__ == '__main__':
    main()

