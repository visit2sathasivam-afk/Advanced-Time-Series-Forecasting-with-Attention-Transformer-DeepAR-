import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_generator import generate_data


def window_data(data, past=30, future=10):
    X, Y = [], []
    for i in range(len(data) - past - future):
        X.append(data[i:i+past])
        Y.append(data[i+past:i+past+future])
    return np.array(X), np.array(Y)


def load_and_preprocess(past=30, future=10):
    data = generate_data()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, Y = window_data(data_scaled, past, future)

    split = int(0.8 * len(X))
    return X[:split], X[split:], Y[:split], Y[split:], scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    print(X_train.shape, y_train.shape)
