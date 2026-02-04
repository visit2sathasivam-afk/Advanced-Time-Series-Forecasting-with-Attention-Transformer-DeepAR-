import numpy as np

def generate_data(n_steps=1500):
    t = np.arange(n_steps)

    trend = t * 0.001
    seasonal = np.sin(0.02 * t)

    s1 = trend + seasonal + np.random.normal(0, 0.1, n_steps)
    s2 = trend * 0.8 + np.sin(0.02 * t + 1) + np.random.normal(0, 0.1, n_steps)
    s3 = trend * 1.2 + np.sin(0.02 * t + 2) + np.random.normal(0, 0.1, n_steps)

    return np.stack([s1, s2, s3], axis=1)


(1500, 3)
