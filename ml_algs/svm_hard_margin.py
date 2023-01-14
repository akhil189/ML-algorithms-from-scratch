import numpy as np

class SVM():
    """Implements hard-margin SVM(Support Vector Machines) classification algorithm.
    
    Learns the parameters using Hinge-loss and Gradient Descent optimization
    
    Parameters
    ----------
    learning_rate : The learning rate for gradient descent.

    lambda_: The regularization parameter.

    n_iterations: The maximum number of iterations for gradient descent.

    Attributes
    ----------
    w : The weights of the SVM classifier.
    
    b : The bias of the SVM classifier.
    """

    def __init__(self, learning_rate: float = 0.0001,
                       lambda_: float = 0.001,
                       n_iterations: float = 1000):
        """Inits the SVM class."""

        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.n_iterations = n_iterations
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the SVM classifier using gradient descent.
        
        Args:
            X: The Data matrix

            y: The target label vector with class labels as +1 and -1.
        
        Returns:
            None
        """

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(self.w, x_i) + self.b)
                if condition >= 1:
                    self.w = self.w - self.learning_rate * (2 * self.lambda_ * self.w)
                else:
                    self.w = self.w - self.learning_rate * (2 * self.lambda_ * self.w  - np.dot(x_i, y[idx]))
                    self.b = self.b - self.learning_rate * (-y[idx])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the class labels for the given data.
        
        Args:
            X: The Data matrix.

        Returns:
            The predicted label vector. {-1, +1}
        """

        pred_ = np.dot(self.w, X) + self.b
        return np.sign(pred_)