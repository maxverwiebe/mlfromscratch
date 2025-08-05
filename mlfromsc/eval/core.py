import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2) # sum of squares of residuals / errors
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) # total sum of squares
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0