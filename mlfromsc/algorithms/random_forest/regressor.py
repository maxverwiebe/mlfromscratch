import numpy as np
from collections import Counter

from ..decision_tree import DecisionTreeRegressor

class RandomForestRegressor:
    def __init__(self, n_trees=200, max_depth=None, min_samples_split=2, bootstrap=True, max_samples=None, random_state=None, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)
        self.trees = []
        self.feature_sets = []

        self.max_features = max_features

    def _bootstrap_idx(self, n):
        if not self.bootstrap:
            k = n
        elif self.max_samples is None:
            k = n
        elif isinstance(self.max_samples, float):
            k = max(1, int(np.ceil(self.max_samples * n)))
        else:
            k = int(self.max_samples)
        return self.rng.integers(0, n, size=k)  # with replacement (zurücklegen)

    def _max_features(self, p):
        if self.max_features is None:
            return p
        elif isinstance(self.max_features, float):
            return max(1, int(np.ceil(self.max_features * p)))
        else:
            return int(self.max_features)

    def fit(self, X, y):
        n, p = X.shape
        self.trees = []
        self.feature_sets = []

        for _ in range(self.n_trees):
            idx = self._bootstrap_idx(n) # picks a set of indices with replacement
            max_feats = self._max_features(p) # number of features to use for this tree
            feats = self.rng.choice(p, max_feats, replace=False) # select features (the selected features will be distinct from each other)
            # so we have random sequences of feats: [11 16  6  9  1  2  5 17  8 15 12  4 14  7  3 10  0 13]
            # [ 7 11  9  5  6 10  3  1 14 15  2  0 13 12  8  4 16 17]
            # [ 1 13 14 10  6  8 15 16 11  4  9 12  2  5  0 17  7  3] ...

            Xs, ys = X[idx][:, feats], y[idx] # select the samples and features for this tree

            print(feats)

            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(Xs, ys)
            self.trees.append(tree)
            self.feature_sets.append(feats)

        return self

    def predict(self, X):
        # predict the target for X by averaging predictions from all trees

        n_samples = X.shape[0]
        all_predictions = np.zeros((self.n_trees, n_samples))
        
        for i, (tree, features) in enumerate(zip(self.trees, self.feature_sets)):
            # select the features used for this tree and predict
            X_subset = X[:, features]
            all_predictions[i] = tree.predict(X_subset)
        
        # return the average prediction across all trees
        return np.mean(all_predictions, axis=0)
