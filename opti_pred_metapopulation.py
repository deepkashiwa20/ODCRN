import scipy.sparse as ss
import numpy as np
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import time
import japanize_matplotlib # pip install japanize-matplotlib
import pyswarms as ps
import math

sigma = 0.2 
gamma = 0.1
migration = 0.1

def SEIR_Single(df, df_flow, beta):
    df_flow = df_flow.apply(lambda x: x/sum(x), axis=1)
    for i in range(len(df_flow)):
        df_flow.iloc[i,i] = 0
    df_out = df_flow.apply(lambda x: sum(x), axis=1)
    df_dec = df.apply(lambda x: (x*df_out), axis=0)
    
    df['N'] = df['S'] + df['E'] + df['I'] + df['R']
    df['Beta'] = beta
    df['S'] = df['S'] + -df['Beta']*df['I']*df['S']/df['N']
    df['E'] = df['E'] + df['Beta']*df['I']*df['S']/df['N'] - sigma*df['E']
    df['I'] = df['I'] + sigma*df['E'] - gamma*df['I']
    df['R'] = df['R'] + gamma * df['I']
    df = df.drop(columns=['N', 'Beta'])
    
    df_inc = pd.DataFrame(np.matmul(df_flow.T.values, df.values), index=TARGET_AREA, columns=['S', 'E', 'I', 'R'])
    df = df + (df_inc - df_dec) * migration
    return df

def SEIR_Multi(beta_list, day_list, df_seir_initial, trans):
    df_seir = df_seir_initial.copy()
    prediction = []
    for i in day_list:
        df_flow = pd.DataFrame(trans[i, :, :], index=TARGET_AREA, columns=TARGET_AREA)
        df_seir = SEIR_Single(df_seir, df_flow, beta_list)
        df_seir_r = df_seir.round().astype(int)
        prediction.append(df_seir_r['I'].tolist())
    return df_seir, np.array(prediction)

def loss(prediction, ground_truth):
    ground_truth = ground_truth.clip(min=1)
    mape_lambda = 300
    return np.mean(np.abs(prediction-ground_truth)) + np.mean(np.abs(prediction-ground_truth)/ground_truth) * mape_lambda

def opt_func(X, day_list, df_seir, trans, infec):
    n_particles = X.shape[0]  # number of particles
    ground_truth = infec[day_list, :]
    dist = []
    for i in range(n_particles):
        _, prediction = SEIR_Multi(X[i, :], day_list, df_seir, trans)
        dist.append(loss(prediction, ground_truth))
    return np.array(dist)

def run_interval_PSO(day_list, df_seir, trans, infec):
    # print('Start Particle Swarm Optimization ...', time.ctime())
    cpu_num = 10
    swarm_size = 50
    iters_num = 100
    beta_low = 0.01
    beta_high = 0.4
    dim = len(TARGET_AREA)  # Dimension of X
    options = {'c1': 0.5, 'c2':0.5, 'w':0.5}
    constraints = (np.array([beta_low] * dim), np.array([beta_high] * dim))
    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dim, options=options, bounds=constraints)
    cost, pos = optimizer.optimize(opt_func, iters=iters_num, n_processes=cpu_num, verbose=False, 
                                   day_list=day_list, df_seir=df_seir, trans=trans, infec=infec)
    return cost, pos

def run_all_PSO(trans, infec):
    assert len(trans) == len(infec), 'len(trans) == len(infec) should be equal'
    T_ALL = len(trans)
    df_seir = pd.DataFrame(np.zeros((len(TARGET_AREA), len(['S', 'E', 'I', 'R'])), dtype='int'), index=TARGET_AREA, columns=['S', 'E', 'I', 'R'])
    df_seir['I'] = infec[0, :]
    df_seir['S'] = POPULATION
    day_list_all = np.arange(T_ALL)
    predictions = []
    opt_betas = [] 
    for i in range(math.ceil(T_ALL / T_INTERVAL)):
        day_list = day_list_all[i*T_INTERVAL:(i+1)*T_INTERVAL]
        cost, pos = run_interval_PSO(day_list, df_seir, trans, infec)
        df_seir, prediction = SEIR_Multi(pos, day_list, df_seir, trans)
        print(i, day_list, cost, pos, time.ctime())
        predictions.extend(prediction)
        opt_betas.append(pos)
    predictions = np.array(predictions)
    opt_betas = np.array(opt_betas)
    df_seir.to_csv(BASE + '/{}_lastseir_{}days.csv'.format(NAME, T_INTERVAL), index=True, header=True)
    np.savetxt(BASE + '/{}_predictions_{}days.csv'.format(NAME, T_INTERVAL), predictions, delimiter=',', fmt='%d')
    np.savetxt(BASE + '/{}_betas_{}days.csv'.format(NAME, T_INTERVAL), opt_betas, delimiter=',', fmt='%.8f')
    print('final predictions...', predictions.shape)
    print('final opt_betas...', opt_betas.shape)
    return predictions, opt_betas, df_seir

#################################################################
POPULATIONPATH = 'data/pref_population_2015.csv'
INFECTIONPATH = 'data/acc_infection_num.csv'
ODPATH = 'data/od_day20180101_20210228.npz'
OD_DAYS = [date.strftime('%Y-%m-%d') for date in pd.date_range(start='2018-01-01', end='2021-02-28', freq='1D')]
BASE = 'MetaPopulationResult_test'
TARGET_AREA = ['東京都', '千葉県','神奈川県','埼玉県']
TARGET_AREA_EN = ['Tokyo', 'Chiba', 'Kanagawa', 'Saitama']
NAME = 'kanto4prefecture_metapopulation'

# TARGET_AREA = ['大阪府', '京都府', '兵庫県']
# TARGET_AREA_EN = ['Osaka', 'Kyoto', 'Hyogo']
# NAME = 'kansai3prefecture_metapopulation'
# START_DATE, END_DATE = '2020-02-03', '2021-01-31'

if len(TARGET_AREA) == 0:
    pref_population = pd.read_csv(POPULATIONPATH)
    TARGET_AREA = pref_population['ken'].values
    POPULATION = pref_population['population'].values
    TARGET_AREA_ID = np.arange(POPULATION).tolist()
    print(POPULATION)
    print(TARGET_AREA_ID)
else:
    pref_population = pd.read_csv(POPULATIONPATH)
    pref_population['gid'] -= 1
    pref_population = pref_population[pref_population['ken'].isin(TARGET_AREA)]
    pref_population = pref_population[['ken', 'gid', 'population']]
    pref_population = pref_population.set_index('ken')
    pref_population = pref_population.reindex(index=TARGET_AREA)
    POPULATION = pref_population['population'].values.squeeze()
    TARGET_AREA_ID = pref_population['gid'].values.squeeze().tolist()
    print(POPULATION)
    print(TARGET_AREA_ID)

START_DATE, END_DATE = '2020-01-20', '2021-01-31'
train_infection = pd.read_csv(INFECTIONPATH)
train_infection['Day'] = pd.to_datetime(train_infection['Day'])
train_infection = train_infection[(train_infection['Day']>=START_DATE) & (train_infection['Day']<=END_DATE)]
train_infection = train_infection.reset_index(drop=True)
train_infection = train_infection.drop(columns=['Day'])
train_infection = train_infection[TARGET_AREA]
train_infection = train_infection.values
print(train_infection.shape)

train_trans = ss.load_npz(ODPATH)
train_trans = np.array(train_trans.todense()).reshape((-1, 47, 47))
train_trans = train_trans[OD_DAYS.index(START_DATE):OD_DAYS.index(END_DATE)+1,:,:]
train_trans = train_trans[:, TARGET_AREA_ID, :][:, :, TARGET_AREA_ID]
print(train_trans.shape)

PRED_START_DATE, PRED_END_DATE = '2021-02-01', '2021-02-28' # The last four weeks.
test_infection = pd.read_csv(INFECTIONPATH)
test_infection['Day'] = pd.to_datetime(test_infection['Day'])
test_infection = test_infection[(test_infection['Day']>=PRED_START_DATE) & (test_infection['Day']<=PRED_END_DATE)]
test_infection = test_infection.reset_index(drop=True)
test_infection = test_infection.drop(columns=['Day'])
test_infection = test_infection[TARGET_AREA]
test_infection = test_infection.values
print(test_infection.shape)

test_trans = ss.load_npz(ODPATH)
test_trans = np.array(test_trans.todense()).reshape((-1, 47, 47))
test_trans = test_trans[OD_DAYS.index(PRED_START_DATE):OD_DAYS.index(PRED_END_DATE)+1,:,:]
test_trans = test_trans[:, TARGET_AREA_ID, :][:, :, TARGET_AREA_ID]
print(test_trans.shape)

T_INTERVAL = 7
###################################################################
def metric(prediction, ground_truth):
    RMSE = np.sqrt(np.mean((prediction-ground_truth)**2))
    MAE = np.mean(np.abs(prediction-ground_truth))
    MAPE = np.mean(np.abs(prediction-ground_truth)/ground_truth)
    return RMSE, MAE, MAPE

def predict(trans, infec):
    betas = np.loadtxt(BASE + '/{}_betas_{}days.csv'.format(NAME, T_INTERVAL), delimiter=',')
    beta = betas[-1, :]
    df_seir = pd.read_csv(BASE + '/{}_lastseir_{}days.csv'.format(NAME, T_INTERVAL), index_col=0)
    _, prediction = SEIR_Multi(beta, np.arange(len(trans)), df_seir, trans)
    np.savetxt(BASE + '/{}_prediction_real_20210201to20210228.csv'.format(NAME), prediction, delimiter=',', fmt='%d')
    print('trans, infections, predictions...', trans.shape, infec.shape, prediction.shape)
    print('RMSE, MAE, MAPE', metric(prediction, infec))
    return prediction

def train():
    predictions, opt_betas, df_seir = run_all_PSO(trans=train_trans, infec=train_infection)
    
def test():
    predictions_real = predict(trans=test_trans, infec=test_infection)

def main():
    train()
    test()
    
if __name__ == '__main__':
    main()