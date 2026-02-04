import tensorflow as tf
from preprocess import load_and_preprocess
from transformer_model import build_transformer


X_train, X_test, y_train, y_test, _ = load_and_preprocess()

model = build_transformer(X_train.shape[1:], y_train.shape[1])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

model.save("transformer_model.keras")