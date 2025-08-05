import numpy as np

# calculates entropy
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

# calculates information gain
def info_gain(y, y_left, y_right):
    H = entropy(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)
    w_left = len(y_left) / len(y)
    w_right = len(y_right) / len(y)
    return H - (w_left * H_left + w_right * H_right)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature # feature by which is splitted
        self.threshold = threshold # threshold
        self.left = left # left child
        self.right = right # right child
        self.value = value # leave value 

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stop condition
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # best split variables
        best_gain = -1
        split_index, split_thresh = None, None

        # test for every feature
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            
            # and every threshold
            for thresh in thresholds:
                left_indexes = X[:, feature_index] <= thresh
                right_indexes = X[:, feature_index] > thresh

                if sum(left_indexes) == 0 or sum(right_indexes) == 0:
                    continue

                y_left, y_right = y[left_indexes], y[right_indexes]
                gain = info_gain(y, y_left, y_right)

                # how good a split at this featrue with threshold would be
                # <=> tries to find the highest info gain
                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_thresh = thresh

        if best_gain == -1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # do the actual split
        # <=> the current node is split using the best feature and threshold found
        left_indexes = X[:, split_index] <= split_thresh
        right_indexes = X[:, split_index] > split_thresh
        left = self._grow_tree(X[left_indexes], y[left_indexes], depth + 1)
        right = self._grow_tree(X[right_indexes], y[right_indexes], depth + 1)
        
        return Node(split_index, split_thresh, left, right)

    def _most_common_label(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        indent = "  " * depth
        if node.value is not None:
            print(f"{indent}Leaf: value = {node.value:.3f}")
        else:
            print(f"{indent}[X{node.feature} <= {node.threshold:.3f}]")
            print(f"{indent}├─ True:")
            self.print_tree(node.left, depth + 1)
            print(f"{indent}└─ False:")
            self.print_tree(node.right, depth + 1)