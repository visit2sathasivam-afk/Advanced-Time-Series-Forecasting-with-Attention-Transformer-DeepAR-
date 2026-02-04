from sklearn.preprocessing import MinMaxScaler

def window_data(data, past=30, future=10):
    X, Y = [], []
    for i in range(len(data)-past-future):
        X.append(data[i:i+past])
        Y.append(data[i+past:i+past+future])
    return np.array(X), np.array(Y)
