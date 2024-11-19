import numpy as np
from collections import Counter
import concurrent.futures
import time

"""
We build on top of the Decision Tree that we implemented earlier.
Instead of only using one decision tree, we can utilize the power
of many trees to build a forest. Through majority voting, we combine 
predictions from all trees (Ensemble Prediction). This method 
helps reduce overfitting and improve generalization.

Bootstrap Aggregation (Bagging):
For each tree, we sample with replacement from the training data,
creating different training sets for each tree, increasing diversity.

Another neat feature is that we can use Python's concurrent future
to parallel process many trees simultaneously.
"""

class DecisionTree:
    # Helper class to create our nodes in the decision tree
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

        def is_leaf_node(self):
            return self.value is not None
    
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _best_criteria(self, X, y, feat_idxs):
        split_idx, split_thresh = None, None
        best_gain = -1
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            
            # For numerical features, using multiple potential thresholds
            # is more robust to outliers by allowing the exploration of 
            # multiple potential split points 
            if np.issubdtype(X_column.dtype, np.number):
                # We use percentiles for thresholds instead of just the mean
                thresholds = np.percentile(X_column, [25, 50, 75])
            else:
                thresholds = np.unique(X_column)
            
            # Calculate the information gain for each threshold
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _grow_tree(self, X, y, depth=0):
        n_labels = len(np.unique(y))
        n_samples, n_features = X.shape

        if (depth >= self.max_depth or n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        
        # If we couldn't find a valid split, return a leaf node
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        
        # If one of the splits would be empty, return a leaf node
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return self.Node(best_feat, best_thresh, left, right)

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None, n_jobs=-1):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.n_jobs = n_jobs if n_jobs > 0 else None  # None means use all available processors
        self.trees = []

    def fit(self, X, y):
        """Fits the random forest to the training data using parallel processing"""
        self.trees = []
        tree_params = {
            'min_samples_split': self.min_samples_split,
            'max_depth': self.max_depth,
            'n_feats': self.n_feats
        }
        
        if self.n_jobs != 1:
            # Parallel processing using concurrent.futures
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all tree fitting tasks
                futures = [
                    executor.submit(_fit_tree, tree_params, X, y)
                    for _ in range(self.n_trees)
                ]
                # Collect results as they complete
                self.trees = [future.result() for future in concurrent.futures.as_completed(futures)]
        else:
            # Sequential processing
            self.trees = [_fit_tree(tree_params, X, y) for _ in range(self.n_trees)]
        
        return self

    def predict(self, X):
        """Predict class for X using majority voting among all trees"""
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        predictions = [self._most_common_label(pred) for pred in tree_predictions]
        return np.array(predictions)
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
def _fit_tree(tree_params, X, y):
    """Helper function to fit a single decision tree."""
    n_samples = X.shape[0]
    
    # Bootstrap sampling
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    X_sample = X[idxs]
    y_sample = y[idxs]
    
    # Create and train tree
    tree = DecisionTree(
        min_samples_split=tree_params['min_samples_split'],
        max_depth=tree_params['max_depth'],
        n_feats=tree_params['n_feats']
    )
    tree.fit(X_sample, y_sample)
    return tree

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    shuffle_index = np.random.permutation(len(X))
    test_index = int(len(X) * test_size)
    X_train = X[shuffle_index[test_index:]]
    y_train = y[shuffle_index[test_index:]]
    X_test = X[shuffle_index[:test_index]]
    y_test = y[shuffle_index[:test_index]]
    return X_train, X_test, y_train, y_test 


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Testing with some generated sample data
    print("Generating sample data...")
    X, y = make_classification(
        n_samples=10000, 
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=1234
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    # Test sequential processing
    print("\nTraining with single process...")
    rf_sequential = RandomForest(n_trees=10, max_depth=10, n_jobs=1)
    start_time = time.time()
    rf_sequential.fit(X_train, y_train)
    sequential_time = time.time() - start_time
    predictions_sequential = rf_sequential.predict(X_test)
    acc_sequential = accuracy(y_test, predictions_sequential)
    print(f"Sequential training time: {sequential_time:.2f} seconds")
    print(f"Sequential accuracy: {acc_sequential:.4f}")
    
    # Test parallel processing
    print("\nTraining with parallel processing...")
    rf_parallel = RandomForest(n_trees=10, max_depth=10, n_jobs=-1)
    start_time = time.time()
    rf_parallel.fit(X_train, y_train)
    parallel_time = time.time() - start_time
    predictions_parallel = rf_parallel.predict(X_test)
    acc_parallel = accuracy(y_test, predictions_parallel)
    print(f"Parallel training time: {parallel_time:.2f} seconds")
    print(f"Parallel accuracy: {acc_parallel:.4f}")
    
    # Calculate speedup
    speedup = sequential_time / parallel_time
    print(f"\nSpeedup: {speedup:.2f}x")