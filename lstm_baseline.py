def build_lstm(input_shape, future_steps):
    model = tf.keras.Sequential([
        layers.LSTM(64, return_sequences=False, input_shape=input_shape),
        layers.Dense(future_steps * input_shape[-1]),
        layers.Reshape((future_steps, input_shape[-1]))
    ])
    return model
