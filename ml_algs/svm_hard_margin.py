import numpy as np

class SVM():
    """Implements hard-margin SVM(Support Vector Machines) Classification algorithm.
    
    Learns the parameters using Gradient Descent optimization
    
    Parameters
    ----------
    learningRate : The learning rate for gradient descent.

    lamda: The regularization parameter.

    nIters: The maximum number of iterations for gradient descent.

    Attributes
    ----------
    w : The weights of the SVM classifier.
    
    b : The bias of the SVM classifier.
    """
    def __init__(self, learningRate: float = 0.0001,
                       lamda: float = 0.001,
                       nIters: float = 1000):
        """Inits the SVM class."""

        self.learningRate = learningRate
        self.lamda = lamda
        self.nIters = nIters
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the SVM classifier using gradient descent.
        
        Args:
            X: The Data matrix

            y: The target vector with class labels as +1 and -1.
        """

        nSamples, nFeatures = X.shape

        self.w = np.zeros(nFeatures)
        self.b = 0

        for _ in range(self.nIters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(self.w, x_i) + self.b)
                if condition >= 1:
                    self.w = self.w - self.learningRate * (2 * self.lamda * self.w)
                else:
                    self.w = self.w - self.learningRate * (2 * self.lamda * self.w  - np.dot(x_i, y[idx]))
                    self.b = self.b - self.learningRate * (-y[idx])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the class labels for the given data.
        
        Args:
            X: The Data matrix.

        Returns:
            The predicted class labels.
        """

        pred_ = np.dot(self.w, X) + self.b
        return np.sign(pred_)