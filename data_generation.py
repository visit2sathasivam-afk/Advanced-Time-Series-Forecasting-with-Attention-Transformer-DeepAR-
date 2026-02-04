import numpy as np


def generate_data(n_steps=1500, seed=42):
    np.random.seed(seed)
    t = np.arange(n_steps)

    trend = 0.001 * t
    seasonal = np.sin(2 * np.pi * t / 50)

    s1 = trend + seasonal + np.random.normal(0, 0.1, n_steps)
    s2 = 0.8 * trend + np.sin(2 * np.pi * t / 50 + 0.5) + np.random.normal(0, 0.1, n_steps)
    s3 = 1.2 * trend + np.sin(2 * np.pi * t / 50 + 1.0) + np.random.normal(0, 0.1, n_steps)

    data = np.stack([s1, s2, s3], axis=1)
    return data


if __name__ == "__main__":
    data = generate_data()
    print("Generated data shape:", data.shape)
    print("First row:", data[0])
