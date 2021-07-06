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
import Metrics
from STGCN12 import *
from Param import *
from Param_STGCN12 import *


def getXSYS_single(allData, mode):
    TRAIN_NUM = int(allData.shape[0] * trainRatio)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = allData[i:i+TIMESTEP_IN, :, :, :]
            y = allData[i+TIMESTEP_IN:i+TIMESTEP_IN+1, :, :, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  allData.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = allData[i:i+TIMESTEP_IN, :, :, :]
            y = allData[i+TIMESTEP_IN:i+TIMESTEP_IN+1, :, :, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    if MODELNAME == 'STGCN12' or MODELNAME == 'STGCN':
        XS = XS.transpose(0, 2, 3, 4, 1)
        XS = XS.reshape(XS.shape[0], XS.shape[1], XS.shape[2], -1)
        XS = XS.transpose(0, 2, 3, 1)
        YS = YS.transpose(0, 2, 3, 4, 1)
        YS = YS.reshape(YS.shape[0], YS.shape[1], YS.shape[2], -1)
        YS = YS.transpose(0, 2, 3, 1)
    return XS, YS

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
    if MODELNAME == 'STGCN12' or MODELNAME == 'STGCN':
        XS = XS.transpose(0, 2, 3, 4, 1)
        XS = XS.reshape(XS.shape[0], XS.shape[1], XS.shape[2], -1)
        XS = XS.transpose(0, 2, 3, 1)
        YS = YS.transpose(0, 2, 3, 4, 1)
        YS = YS.reshape(YS.shape[0], YS.shape[1], YS.shape[2], -1)
        YS = YS.transpose(0, 2, 3, 1)
    return XS, YS

def getModel(name):
    if name == 'STGCN12':
        ks, kt, bs, T, n, p = 3, 3, [[47, 32, 64], [64, 32, 128]], 12, 47, 0
        W = np.load(ADJPATH)
        L = scaled_laplacian(W)
        Lk = cheb_poly(L, ks)
        Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
        model = STGCN(ks, kt, bs, T, n, Lk, p).to(device)
        summary(model, (47, 12, 47), device="cuda:{}".format(GPU))
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
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * 0.8)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

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
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())
        
def testModel(name, mode, XS, YS, YS_multi):
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=True)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    criterion = nn.MSELoss()
    torch_score = evaluate_model(model, criterion, test_iter)
    # model.eval()
    # with torch.no_grad():
    #    YS_pred = model(XS_torch).cpu().numpy()
    model.eval()
    XS_pred_multi, YS_pred_multi = [XS_torch], []
    with torch.no_grad():
        for i in range(TIMESTEP_OUT):
            tmp_torch = torch.cat(XS_pred_multi, axis=2)[:, :, i:, :]
            YS_pred = model(tmp_torch)
            print('type(YS_pred), YS_pred.shape, XS_tmp_torch.shape', type(YS_pred), YS_pred.shape, tmp_torch.shape)
            XS_pred_multi.append(YS_pred)
            YS_pred_multi.append(YS_pred)
        YS_pred_multi = torch.cat(YS_pred_multi, axis=2).cpu().numpy()
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred_multi)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS_multi)
    print('YS_multi.shape, YS_pred_multi.shape,', YS_multi.shape, YS_pred_multi.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS_multi, YS_pred_multi)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Testing Ended ...', time.ctime())
        
################# Parameter Setting #######################
MODELNAME = 'STGCN12'
KEYWORD = 'predMultiOD_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M") + '_log_single2multi'
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

def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_STGCN12.py', PATH)
    shutil.copy2('STGCN12.py', PATH)
        
    prov_day_data = ss.load_npz(ODPATH)
    prov_day_data_dense = np.array(prov_day_data.todense()).reshape((-1, 47, 47))
    data = prov_day_data_dense[STARTINDEX:ENDINDEX+1,:,:,np.newaxis]
    data = np.log(data + 1.0)
    print('STARTDATE, ENDDATE', STARTDATE, ENDDATE, 'data.shape', data.shape)

    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS_single(data, 'TRAIN')
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'train', trainXS, trainYS)
    
    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS_single(data, 'TEST')
    _, testYS_multi = getXSYS(data, 'TEST')
    print('TEST XS.shape, YS.shape, YS_multi.shape', testXS.shape, testYS.shape, testYS_multi.shape)
    testModel(MODELNAME, 'test', testXS, testYS, testYS_multi)

    
if __name__ == '__main__':
    main()

