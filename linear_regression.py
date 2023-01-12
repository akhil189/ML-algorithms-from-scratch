import numpy as np
from tqdm import tqdm
from typing import List
from typing import Tuple

class LinearRegression:
    """Implements Linear Regression algorithm.
    
    Parameters
    ----------
    learning_rate : default=0.01
        The learning rate for Gradient Descent optimization.

    batch_size : default=None,
        The batch size to be used in each iteration of Gradient Descent optimization.
        Only used if `optimizer` is 'mbgd' i.e., Mini-Batch Gradient Descent.

    max_iterations : default=5000,
        The maximum number of iterations for gradient descent optimization.

    loss : {'sse', 'mse', 'rmse'}, default='mse'
        The loss function to be used.
    
    penalty : {'l1', 'l2', 'elasticnet', None}, default=None
        The regularization term to be used. By default, no regularization is used.

    lambda_ : float, default=0.0
        The coefficient for regulaization term.
        Only used if penalty is not None.

    tolerance : float, default=1e-3
        The stopping criterion for gradient descent optimization.
        If the difference between two consecutive losses is smaller than `tolerance`,
        the optimization process is stopped.

    optimizer : {'gradient_descent', None}, default=None
        If optimizer is not specified, Closed-Form-Solution is used.

        If `gradient_descent`, the following variant of gradient descent is used
        - `batch_size` = None => Batch Gradient Descent
        - `batch_size` = integer => Mini-Batch Gradient Descent
        - `batch_size` = 1 => Online Gradient Descent

        Note: If 'batch_size' greater than n_samples is given, Batch Gradient Descent is used.

    Attributes
    ----------
    theta : np.ndarray, shape=(n_features,)
        The learned parameters.
    
    errors : list[float],
        The history of errors from each iteration.
    """
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 loss: str = 'mse',
                 optimizer: str = None,
                 batch_size: int = None,
                 max_iterations: int = 5000,                
                 penalty: str = None,
                 lambda_: float = 0.0,
                 tolerance: float = 0.001):
        """Inits LinearRegression class with default parameters."""
        
        losses = {'sse' : self.sse,
                  'mse' : self.mse,
                  'rmse': self.rmse}

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.loss = losses[loss]
        self.tolerance = tolerance
        self.optimizer = optimizer
        self.penalty = penalty
        self.lambda_ = lambda_
    
    def sse(self, y, y_hat) -> float:
        """Computes Sum-Squared Error(OLS - Ordinary Least Squares)."""
        
        sse = ((y-y_hat)**2).sum()
        return sse

    def mse(self, y, y_hat) -> float:
        """Computes Mean-Squared-Error."""
        
        mse = self.sse(y, y_hat)/y.shape[0]
        return mse

    def rmse(self, y, y_hat) -> float:
        """Computes Root-Mean-Squared-Error."""
        
        rmse = np.sqrt(self.mse(y, y_hat))
        return rmse

    def closed_form_solution(self, X, y) -> None:
        """Computes Closed-Form solution using Normal Equations if it's exists.
        
        Normal Equation: (X^T.X)^-1.(X.T.y)
        X^T.X is invertible iff rank(X) = n_features (This forces n_samples >= n_features)
        
        For X.T.X to be invertible, the following should be true
        1. n_samples >= n_features
        2. rank(X) = min(n_samples, n_features)

        We do the above to avoid complex calculation of checking if X^T.X is invertible.
        """

        n_samples, n_features = X.shape

        X_rank = np.linalg.matrix_rank(X)

        if n_samples < n_features:
            raise ValueError("Data matrix is not Full-rank. Closed-Form-Solution does not exist. Try using an optimizer.")

        if X_rank != min(n_samples, n_features):
            raise ValueError("Data matrix is not Full-rank. Closed-Form-Solution does not exist. Try using an optimizer.")
        
        print("Data matrix is Full-rank. Closed-form-solution exists.")
        
        I = np.eye(X.shape[1], dtype=float)
        theta = np.linalg.inv(X.T.dot(X) + self.lambda_*I).dot(X.T).dot(y)
        
        return theta
    
    def cost_derivative(self, X, y, theta):
        """Computes dJ/dw."""
        
        y_hat = X.dot(theta)
        dJ = X.T.dot(y_hat-y) + self.lambda_*theta
        
        return dJ

    def get_batches(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    batch_size: int):
        """Shuffles data and creates batches of `self.batch_size`.
        
        Args:
            X : Data matrix

            y : Target vector

            batch_size : Number of samples per batch
        
        Returns:
            List of batches
        """

        batches = []
        data = np.hstack((X, y[:, np.newaxis]))
        np.random.shuffle(data)
        n_batches = data.shape[0]//batch_size
        
        for i in range(n_batches):
            batch = data[i*batch_size : (i+1)*batch_size, :]
            X, y = batch[:, :-1], batch[:, -1]
            batches.append((X, y))
        if data.shape[0] % batch_size > 0:
            batch = data[n_batches*batch_size : data.shape[0], :] # remaining samples
            X, y = batch[:, :-1], batch[:, -1]
            batches.append((X, y))
        return batches

    def gradient_descent(self, X, y) -> None:
        
        self.errors = []
        theta = np.zeros(X.shape[1], dtype= float)

        prev_error = float('inf')
        stopFlag = False
        for i in tqdm(range(self.max_iterations)):

            if not self.batch_size or self.batch_size > X.shape[0]:
                self.batch_size = X.shape[0] # Full-batch gradient descent
            
            batches = self.get_batches(X, y, batch_size=self.batch_size)

            for batch in batches:
                
                X_batch, y_batch = batch
                theta = theta - (self.learning_rate/X_batch.shape[0])*self.cost_derivative(X_batch, y_batch, theta)
                y_hat = X_batch.dot(theta)
                
                curr_error = self.loss(y_batch, y_hat)
                self.errors.append(curr_error)

                # Stopping criteria
                error_difference = prev_error - curr_error
                if abs(error_difference) < self.tolerance:
                    print(f"The model stopped learning - Converged in {i} steps(iterations)")
                    stopFlag = True
                    break
                prev_error = curr_error

            if stopFlag:
                break

        return theta
        
    def fit(self,
            X: np.ndarray,
            y: np.ndarray) -> None:
        """Learn the parameters of the model i.e., `theta`. 
        
        Args:
            X : Data matrix

            y : Target vector
        """

        n_samples = X.shape[0]
        X_train = np.column_stack( (np.ones(shape=(n_samples,1)), X) )

        if not self.optimizer:
            self.theta = self.closed_form_solution(X_train, y)
        else:
            self.theta = self.gradient_descent(X_train, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Computes predictions.
        
        Args:
            X : Data matrix

        Returns:
            y_hat: Predictions
        """

        n_samples = X.shape[0]
        X_test = np.column_stack( (np.ones(shape=(n_samples, 1)), X) )

        y_hat =  X_test.dot(self.theta)
        return y_hat