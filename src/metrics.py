import numpy as np
import pandas as pd

def sMAPE(yhat, y):
    return 2 * abs(yhat - y) / (abs(y) + abs(yhat))

def MAE(yhat, y):
    return np.abs(yhat - y)

def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))
