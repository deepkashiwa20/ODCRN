import pandas as pd

TIMESTEP_IN, TIMESTEP_OUT = 7, 7
HEIGHT = 47
WIDTH = 47
CHANNEL = 1
BATCHSIZE = 4
SPLIT = 0.2
LEARN = 0.0001
EPOCH = 200
PATIENCE = 10
LOSS = 'mse'
OPTIMIZER = 'adam'
trainRatio = 0.8  # 80 days for training and validation, 20 days for testing.

ODPATH = '/data2/cai/OD/japan/aggregate_prov/results/od_day20180101_20210228.npz'
ADJPATH = '../Adjacency/adjacency_matrix.npy'
OD_DAYS = [date.strftime('%Y-%m-%d') for date in pd.date_range(start='2018-01-01', end='2021-02-28', freq='1D')]
STARTDATE, ENDDATE = '2020-01-01', '2021-02-28'
STARTINDEX, ENDINDEX = OD_DAYS.index(STARTDATE), OD_DAYS.index(ENDDATE)
FILEPATH = f'../workJapanOD_EX1_{STARTDATE}_{ENDDATE}/'