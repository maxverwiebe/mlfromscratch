
# basically a copy of the classifier
# just replaced the "loss" functions

import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def mse(y):
    return np.mean((y - np.mean(y)) ** 2)

def mse_gain(y, y_left, y_right):
    parent_error = mse(y)
    w_left = len(y_left) / len(y)
    w_right = len(y_right) / len(y)
    return parent_error - (w_left * mse(y_left) + w_right * mse(y_right))

class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return Node(value=np.mean(y))

        best_gain = -1
        split_index, split_thresh = None, None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for thresh in thresholds:
                left_indexes = X[:, feature_index] <= thresh
                right_indexes = X[:, feature_index] > thresh

                if sum(left_indexes) == 0 or sum(right_indexes) == 0:
                    continue

                y_left, y_right = y[left_indexes], y[right_indexes]
                gain = mse_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_thresh = thresh

        if best_gain == -1:
            return Node(value=np.mean(y))

        left_indexes = X[:, split_index] <= split_thresh
        right_indexes = X[:, split_index] > split_thresh
        left = self._grow_tree(X[left_indexes], y[left_indexes], depth + 1)
        right = self._grow_tree(X[right_indexes], y[right_indexes], depth + 1)
        return Node(split_index, split_thresh, left, right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)