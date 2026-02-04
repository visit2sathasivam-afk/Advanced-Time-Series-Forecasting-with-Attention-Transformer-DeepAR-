import numpy as np
from transformer_model import build_transformer
from preprocess import load_and_preprocess_data


def tune_transformer(X_train, y_train):
    configs = [
        {"embed": 32, "heads": 2, "ff": 64},
        {"embed": 64, "heads": 4, "ff": 128},
        {"embed": 128, "heads": 4, "ff": 256},
    ]

    best_loss = float("inf")
    best_model = None
    best_config = None

    for cfg in configs:
        print(f"Testing config: {cfg}")

        model = build_transformer(
            input_shape=X_train.shape[1:],
            future_steps=y_train.shape[1]
        )

        model.compile(optimizer="adam", loss="mse")

        history = model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        val_loss = min(history.history["val_loss"])
        print("Validation loss:", val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_config = cfg

    print("\nBest Config:", best_config)
    print("Best Loss:", best_loss)

    return best_model, best_config
