import numpy as np
import tensorflow as tf
from preprocess import load_and_preprocess


def rmse(y, p):
    return np.sqrt(np.mean((y - p) ** 2))


def mae(y, p):
    return np.mean(np.abs(y - p))


def wape(y, p):
    return np.sum(np.abs(y - p)) / np.sum(np.abs(y))


X_train, X_test, y_train, y_test, _ = load_and_preprocess()

t_model = tf.keras.models.load_model("transformer_model.keras")
l_model = tf.keras.models.load_model("lstm_model.keras")

tp = t_model.predict(X_test)
lp = l_model.predict(X_test)

print("Transformer RMSE:", rmse(y_test, tp))
print("Transformer MAE :", mae(y_test, tp))
print("Transformer WAPE:", wape(y_test, tp))

print("LSTM RMSE:", rmse(y_test, lp))
print("LSTM MAE :", mae(y_test, lp))
print("LSTM WAPE:", wape(y_test, lp))
