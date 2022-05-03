import numpy as np



def evaluate(y_pred: np.array, y_true: np.array, precision=10):
    print('MSE:', round(MSE(y_pred, y_true), precision))
    print('RMSE:', round(RMSE(y_pred, y_true), precision))
    print('MAE:', round(MAE(y_pred, y_true), precision))
    print('MAPE:', round(MAPE(y_pred, y_true)*100, precision), '%')
    print('PCC:', round(PCC(y_pred, y_true), precision))
    return MSE(y_pred, y_true), RMSE(y_pred, y_true), MAE(y_pred, y_true), MAPE(y_pred, y_true)

def MSE(y_pred: np.array, y_true: np.array):
    return np.mean(np.square(y_pred - y_true))

def RMSE(y_pred:np.array, y_true:np.array):
    return np.sqrt(MSE(y_pred, y_true))

def MAE(y_pred:np.array, y_true:np.array):
    return np.mean(np.abs(y_pred - y_true))

def MAPE(y_pred:np.array, y_true:np.array, epsilon=1e-3):       # avoid zero division
    return np.mean(np.abs(y_pred - y_true) / np.clip((np.abs(y_pred) + np.abs(y_true)) * 0.5, epsilon, None))

def PCC(y_pred:np.array, y_true:np.array):      # Pearson Correlation Coefficient
    return np.corrcoef(y_pred.flatten(), y_true.flatten())[0,1]


