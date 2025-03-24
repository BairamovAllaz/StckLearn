import numpy as np


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    # for calculating gini Impurity
    def gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def best_slip(self, X, y):
        m, n = X.shape
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        best_left_idx, best_right_idx = None, None

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = np.where(X[:, feature] <= threshold)[0]
                right_idx = np.where(X[:, feature] > threshold)[0]
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                gini = (len(left_idx) / m) * self.gini_impurity(y[left_idx]) + \
                    (len(right_idx) / m) * self.gini_impurity(y[right_idx])

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
                    best_left_idx, best_right_idx = left_idx, right_idx
        return best_feature, best_threshold, best_left_idx, best_right_idx

    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:  # Pure class
            return Node(value=y[0])
        if self.max_depth and depth >= self.max_depth:  # Max depth reached
            most_common = np.bincount(y).argmax()
            return Node(value=most_common)
        feature, threshold, left_idx, right_idx = self.best_split(X, y)
        if feature is None:
            most_common = np.bincount(y).argmax()
            return Node(value=most_common)
        left_subtree = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self.build_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(feature, threshold, left_subtree, right_subtree)
    


    def fit(self,X,y):
        self.root = self.build_tree(X, y)

    def predict_sample(self, node, x):
        if node.value is not None:
            return node.value  # Return class label
        if x[node.feature] <= node.threshold:
            return self.predict_sample(node.left, x)
        return self.predict_sample(node.right, x)

    def predict(self, X):
        return np.array([self.predict_sample(self.root, x) for x in X])
    
    
