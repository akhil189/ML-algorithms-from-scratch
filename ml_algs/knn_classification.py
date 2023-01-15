import numpy as np
from collections import Counter

def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Computes Euclidean distance between two vectors."""

    return np.sqrt(np.sum((x1 - x2)**2))

class KNNClassifier():

    def __init__(self, k=3):
        """Inits the KNNClassifier."""
        
        self.k = k
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Saves the training data and labels.
        
        Args:
            X : Training data.

            y: Training labels.
        
        Returns:
            None
        """

        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the class labels for the provided data.
        
        Args:
            X: The data matrix, shape = (n_samples, n_features).

        Returns:
            predictions: The predicted class labels for each data matrix, shape = (n_samples).
        """
        
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x: np.ndarray) -> int:
        """Predicts the class labels for each sample/datapoint.
        
        Args:
            x : A data point/sample.
        
        Returns:
            prediction: The predicted class label.
        """

        # Computing distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Indices of k nearest data points
        k_nearest_indices = np.argsort(distances)[:self.k]
        
        # Lables of k nearest data points
        k_nearest_labels = [self.y_train[idx] for idx in k_nearest_indices]
        
        # Voting
        prediction = max(k_nearest_labels, key = k_nearest_labels.count)
        
        ## Using Counter() method
        # most_common = Counter(k_nearest_labels).most_common(1)
        # prediction = most_common[0][0]

        return prediction