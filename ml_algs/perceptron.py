import numpy as np

class Perceptron():
    """Implements the Perceptron algorithm."""

    def __init__(self,
                 learning_rate: float = 0.01,
                 n_iters: int = 1000):
        """Inits the Perceptron class."""

        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activation_fn = self.unit_step_function
        self.w = None
        self.b = None

    def unit_step_function(self, x: np.ndarray) -> np.ndarray:
        """Computes the step function values."""

        return np.where(x >= 0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Learns the parameters using Perceptron algorithm.
        
        Args:
            X : Training data, shape = (n_samples, n_features).

            y: Target values, shape = (n_samples, ).
        
        Returns:
            None
        """
        
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0

        y_true = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):

            for idx, x in enumerate(X):
                
                linear_output = np.dot(self.w, x) + self.b
                y_pred = self.activation_fn(linear_output)

                if y_true[idx] != y_pred:
                    self.w = self.w + self.learning_rate * (y_true[idx] - y_pred) * x
                    self.b = self.b + self.learning_rate * (y_true[idx] - y_pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts labels for the given samples.
        
        Args:
            X: Data matrix, shape = (n_samples, n_features).

        Returns:
            y_pred: Predicted labels, shape = (n_samples, ).
        """

        linear_output = X.dot(self.w) + self.b
        y_pred = self.activation_fn(linear_output)
        return y_pred