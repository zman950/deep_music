import numpy as np


def get_pre_processed_data(filepath="deep_music/data/interim/samples.npy"):
    pre_processed_data = np.load(filepath)
    X = pre_processed_data[:, :60]
    y = pre_processed_data[:, 60:96]
    return X, y

if __name__ == "__main__":
    X, y = get_pre_processed_data()
    print(X.shape)
