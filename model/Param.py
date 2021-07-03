import pandas as pd

TIMESTEP_IN, TIMESTEP_OUT = 7, 7
HEIGHT = 47
WIDTH = 47
CHANNEL = 1
BATCHSIZE = 4
SPLIT = 0.2 # Used for Keras.
LEARN = 0.0001
EPOCH = 200
PATIENCE = 20
LOSS = 'mae'
OPTIMIZER = 'adam'
TRAINRATIO = 0.8
VALIDRATIO = 0.8 # 0.8 * 0.8 = 0.64 as training data and 0.16 as validation data # Used for Torch.
ODPATH = '../data/od_day20180101_20210228.npz'
ADJPATH = '../data/adjacency_matrix.npy'
OD_DAYS = [date.strftime('%Y-%m-%d') for date in pd.date_range(start='2018-01-01', end='2021-02-28', freq='1D')]
STARTDATE, ENDDATE = '2020-01-01', '2021-02-28'
STARTINDEX, ENDINDEX = OD_DAYS.index(STARTDATE), OD_DAYS.index(ENDDATE)
FILEPATH = f'../save/'
