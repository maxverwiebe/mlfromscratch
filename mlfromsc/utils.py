import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=0):
    """Split the dataset into training and testing sets."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    test_sz = int(len(X) * test_ratio)
    test_idx = indices[:test_sz]
    train_idx = indices[test_sz:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]