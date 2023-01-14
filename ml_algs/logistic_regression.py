import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from typing import Tuple

class LogisticRegression:
    """Implements Logistic Regression algorithm.

    Parameters
    ----------
        learning_rate: The learning rate for gradient descent, default is 0.001.

        max_iterations: The maximum number of iterations, default is 5000.

        tolerance: The tolerance of convergence, default is 1e-3.

    Attributes
    ----------
        theta: np.ndarray, shape=(n_features+1, ) The learnt parameters.

        errors: list[float], The history of error sequence from gradient descent.
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 max_iterations: int = 5000,
                 tolerance: float = 0.001):
        """Inits LogisticRegression class with default parameters."""

        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function."""

        sig = 1/( 1+np.exp(-z) )
        return sig

    def cost_function(self, X, y, theta) -> float:
        """Computes cost function.
        
        Args:
            X: The data matrix, shape = (n_samples, n_features + 1).

            y: The target vector, shape = (n_samples, 1).

            theta: The learned parameters, shape = (n_features + 1, 1).
        
        Returns:
            J: The cost function value.
        """

        n_samples = X.shape[0]
        
        """approach 1: Minimizing NLL"""
        pred_ = np.log(np.ones(X.shape[0]) + np.exp(X.dot(theta))) - X.dot(theta)*y # Negative Log-likelihood
        J = (1/n_samples)*pred_.sum()

        """approach 2: Minimizing Cross-Entropy"""
        # y_hat = self.sigmoid(X.dot(theta))
        # pred_ = y * np.log(y_hat) + (1-y) * np.log(1-y_hat)
        # J = (-1/n_samples)*pred_.sum()

        return J

    def cost_derivative(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Computes dJ/dw i.e, derivative of the cost function(NLL) w.r.to theta.
        
        Args:
            X : The Data matrix

            y : The target vector

            theta : The parameters of the model
            
        Returns:
            dJ : The derivative of the cost function(NLL) w.r.to theta, shape = (n_features, 1)
        """

        n_samples = X.shape[0]
        y_hat = self.sigmoid(X.dot(theta))
        dJ = (1 / n_samples) * X.T.dot(y_hat - y)
        
        return dJ
  
    def gradient_descent(self, X, y) -> np.ndarray:

        """Performs gradient descent.
        
        Args:
            X: The data matrix, shape = (n_samples, n_features + 1).

            y: The target vector, shape = (n_samples, 1).
        
        Returns:
            theta: The learned parameters, shape = (n_features + 1, 1).
        """

        n_features = X.shape[1]
        theta = np.zeros(n_features, dtype= np.float64)

        self.errors = []
        prev_error = float('inf')

        for i in tqdm(range(self.max_iterations)):
            
            theta = theta - self.learning_rate * self.cost_derivative(X, y, theta)
            
            curr_error = self.cost_function(X, y, theta)
            self.errors.append(curr_error)

            # Stopping criteria
            error_diff = prev_error - curr_error
            if abs(error_diff) < self.tolerance:
                print("The model stopped - No Further Improvment")
                break    
            prev_error = curr_error

        return theta

    def fit(self, X, y) -> None:
        """Learns the model parameters through gradient descent.

        Args:
            X: The data matrix, shape = (n_samples, n_features).

            y: The target vector, shape = (n_samples, 1).
        
        Returns:
            None.
        """
        n_samples = X.shape[0]
        X_train = np.column_stack( (np.ones(shape=(n_samples, 1)), X) )

        self.theta = self.gradient_descent(X_train, y)
    
    def predict(self, X) -> np.ndarray:
        """Predicts the target labels.

        Args:
            X: The data matrix, shape = (n_samples, n_features).

        Returns:
            The predicted labels, shape = (n_samples, ).
        """
        n_samples = X.shape[0]
        X_test = np.column_stack( (np.ones(shape=(n_samples, 1)), X) )
        y_hat = self.sigmoid(X_test.dot(self.theta)) # X.dot(theta) is same as theta.T.dot(X)    
        
        return np.around(y_hat)

    def evaluate(self, y, y_hat) -> Tuple[float, float, float]:
        """Computes Accuracy, Precision and Recall.

        Args:
            y: The target vector, shape = (n_samples, 1).
            
            y: The predicted label vector, shape = (n_samples, 1).
        
        Returns:
            A Tuple containing the accuracy, precision and recall.
        """

        y = (y == 1) # same as y.sum()
        y_hat = (y_hat == 1)

        accuracy = (y == y_hat).sum() /y.size
        precision = (y & y_hat).sum() /y_hat.sum()
        recall = (y & y_hat).sum() / y.sum()

        return accuracy, precision, recall

    def plotCost(self) -> None:
        """Plots the cost sequence from gradient descent.
        Args:
            None
            
        Returns:
            None
        """

        s = np.array(self.errors)
        t = np.arange(s.size)

        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(xlabel = 'iterations', ylabel = 'cost', title = 'cost trend')
        ax.grid()

        plt.legend(bbox_to_anchor = (1.05, 1), shadow=True)
        plt.show()