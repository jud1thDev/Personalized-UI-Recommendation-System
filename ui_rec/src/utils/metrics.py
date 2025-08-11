from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import numpy as np


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def auc(y_true, y_prob) -> float:
    return float(roc_auc_score(y_true, y_prob))


def acc(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred)) 