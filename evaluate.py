import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def mase(y_true, y_pred, y_train):
    naive_forecast = y_train[1:] - y_train[:-1]
    scale = np.mean(np.abs(naive_forecast))
    return np.mean(np.abs(y_true - y_pred)) / scale


y_pred = model.predict(X_test)

print("RMSE:", rmse(y_test, y_pred))
print("MAE :", mae(y_test, y_pred))
print("WAPE:", wape(y_test, y_pred))
