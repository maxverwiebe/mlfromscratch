import numpy as np
from collections import Counter

from ..decision_tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for tree in range(self.n_trees):
            # random rows with replacement
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            # random features: either self.max_features or sqrt(n_features)
            n_features = X.shape[1]
            max_feats = self.max_features or int(np.sqrt(n_features))
            feature_indices = np.random.choice(n_features, max_feats, replace=False)

            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        # ensemble predictions of all trees
        tree_preds = []
        for tree, feats in self.trees:
            preds = tree.predict(X[:, feats])
            tree_preds.append(preds)
        
        # pick the most common prediction
        tree_preds = np.array(tree_preds) # shape: (n_trees, n_samples)
        print(tree_preds)
        majority_votes = []
        
        for i in range(X.shape[0]):
            votes = tree_preds[:, i]# all predictions for sample i
            vote_counts = Counter(votes) # count how often each prediction occurs
            most_common_vote = vote_counts.most_common(1)[0][0] # most common prediction / majority
            majority_votes.append(most_common_vote)
            
        return np.array(majority_votes)