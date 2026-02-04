import tensorflow as tf
from tensorflow.keras import layers
from preprocess import load_and_preprocess


def build_lstm(input_shape, future_steps):
    m = tf.keras.Sequential([
        layers.LSTM(64, input_shape=input_shape),
        layers.Dense(future_steps * input_shape[-1]),
        layers.Reshape((future_steps, input_shape[-1]))
    ])
    return m


X_train, X_test, y_train, y_test, _ = load_and_preprocess()

model = build_lstm(X_train.shape[1:], y_train.shape[1])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

model.save("lstm_model.keras")
