import numpy as np
from tqdm import tqdm
from typing import List
from typing import Tuple

class LinearRegression:
    """Implements Linear Regression algorithm.
    
    Note:
    - Closed-form solution with l1 penalty raises error as there is no analytical solution for lasso regression.
    
    Parameters
    ----------
    loss : {'sse', 'mse', 'rmse'}, default='mse'
        The loss function to be used.
    
    penalty : {'l1', 'l2', None}, default=None
        The regularization term to be used. By default, no regularization is used.

    lambda_ : float, default=0.0
        The coefficient for regulaization term.
        Only used if `penalty` is 'l1' or 'l2'.

    optimizer : {'gradient_descent', None}, default=None
        If `optimizer` is not specified, Closed-Form-Solution is used.

        If 'gradient_descent', the following variant of gradient descent is used
        - `batch_size` = None => Batch Gradient Descent
        - `batch_size` = integer => Mini-Batch Gradient Descent
        - `batch_size` = 1 => Online Gradient Descent

        Note: 
        - If the specified 'batch_size' is greater than n_samples,
          Batch Gradient Descent is used.

    learning_rate : default=0.01
        The learning rate for Gradient Descent optimization.

    batch_size : default=None,
        The batch size to be used in each iteration of Gradient Descent optimization.
        Only used if `optimizer` is 'gradient_descent'

    max_iterations : default=5000,
        The maximum number of iterations for gradient descent optimization.

    tolerance : float, default=1e-3
        The stopping criterion for gradient descent optimization.
        If the difference between two consecutive losses is smaller than `tolerance`,
        the optimization process is stopped.

    Attributes
    ----------
    theta : np.ndarray, shape=(n_features,)
        The learned parameters.
    
    errors : list[float],
        The history of errors from each iteration.
    """
    
    def __init__(self,
                 loss: str = 'mse',
                 penalty: str = None,
                 lambda_: float = 0.0,
                 optimizer: str = None,
                 learning_rate: float = 0.001,
                 batch_size: int = None,
                 max_iterations: int = 5000,                
                 tolerance: float = 0.001):
        """Inits LinearRegression class with default parameters."""
        
        losses = {'sse' : self.sse,
                  'mse' : self.mse,
                  'rmse': self.rmse}

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.loss = loss
        self.loss_function = losses[loss]
        self.tolerance = tolerance
        self.optimizer = optimizer
        self.penalty = penalty
        self.lambda_ = lambda_
    
    def sse(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Computes Sum-Squared Error(OLS - Ordinary Least Squares)."""
        
        sse = ((y-y_hat)**2).sum()
        return sse

    def mse(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Computes Mean-Squared-Error."""
        
        mse = self.sse(y, y_hat)/y.shape[0]
        return mse

    def rmse(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Computes Root-Mean-Squared-Error."""
        
        rmse = np.sqrt(self.mse(y, y_hat))
        return rmse

    def closed_form_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes Closed-Form solution using Normal Equations if it's exists.
        
        Normal Equation: (X^T.X)^-1.(X.T.y)
        X^T.X is invertible iff rank(X) = n_features (This forces n_samples >= n_features)
        
        For X.T.X to be invertible, the following should be true
        1. n_samples >= n_features
        2. rank(X) = min(n_samples, n_features)

        We do the above to avoid complex calculation of checking if X^T.X is invertible.

        Args:
            X : Data Matrix, shape = (n_samples, n_features)

            y : Target Matrix, shape = (n_samples, 1)

        Returns:
            theta : The parameter vector computed using Normal Equation.
        """

        n_samples, n_features = X.shape

        X_rank = np.linalg.matrix_rank(X)

        if n_samples < n_features:
            raise ValueError("Data matrix is not Full-rank. Closed-Form-Solution does not exist. Try using an optimizer.")

        if X_rank != min(n_samples, n_features):
            raise ValueError("Data matrix is not Full-rank. Closed-Form-Solution does not exist. Try using an optimizer.")
        
        print("Data matrix is Full-rank. Closed-form-solution exists.")

        I = np.eye(X.shape[1], dtype=float)
        
        if not self.penalty :
            theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        elif self.penalty == 'l1':
            raise ValueError("No Closed-Form-Solution(Analytic Solution) exists for Lasso regression. Specify an optimizer.")
        elif self.penalty == 'l2':
            theta = np.linalg.inv(X.T.dot(X) + self.lambda_*I).dot(X.T).dot(y)

        return theta
    
    def cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """Computes Cost function as per the specified `loss`
        
        Args:
            X : The Data matrix, shape=(n_samples, n_features+1)

            y : The target vector, shape=(n_samples,)

            theta : The parameter vector, shape=(n_features+1,)

        Returns:
            J : The value of the Cost function computed using the specified `loss`
        """

        if not self.penalty:
            penalty_term = 0
        elif self.penalty == 'l1':
            penalty_term = self.lambda_* np.sum(np.abs(theta))
        elif self.penalty == 'l2':
            penalty_term = 0.5 * self.lambda_ * theta.dot(theta)
        
        y_hat = X.dot(theta)
        J = self.loss_function(y, y_hat) + penalty_term
        
        return J

    def cost_derivative(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Computes gradient of Cost function, dJ/dw as per the `loss`.
        
        Args:
            X : The Data matrix, shape = (n_samples, n_features + 1)

            y : The Target matrix, shape = (n_samples,)

            theta : The parameter vector, shape = (n_features + 1,)
        
        Returns:
            dJ : The gradient of the Cost function w.r.to theta
        """

        if not self.penalty:
            penalty_derivative = 0
        elif self.penalty == 'l1':
            penalty_derivative = self.lambda_*np.sign(theta)
        elif self.penalty == 'l2':
            penalty_derivative = self.lambda_*theta

        y_hat = X.dot(theta)
        sse_loss_derivative = X.T.dot(y_hat - y)
        mse_loss_derivative = 1/(X.shape[0])*(sse_loss_derivative)
        rmse_loss_derivative = 1/(2*np.sqrt(self.mse(y, y_hat)))*mse_loss_derivative

        dJ = None
        if self.loss == 'sse':
            dJ = sse_loss_derivative + penalty_derivative
        elif self.loss == 'mse':
            dJ = mse_loss_derivative + penalty_derivative
        elif self.loss == 'rmse':
            dJ = rmse_loss_derivative + penalty_derivative
        
        return dJ

    def get_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> List:
        """Shuffles data and creates batches of `self.batch_size`.
        
        Args:
            X : Data matrix

            y : Target vector

            batch_size : Number of samples per batch
        
        Returns:
            batches: List of batches
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
        """Performs gradient descent optimization.

        Args:
            X : The Data Matrix, shape = (n_samples, n_features+1)

            y: The target vector, shape = (n_samples,)

        Returns:
            theta : The parameter vector, shape = (n_features+1,)
        """
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
                theta = theta - (self.learning_rate)*self.cost_derivative(X_batch, y_batch, theta)
                y_hat = X_batch.dot(theta)
                
                curr_error = self.loss_function(y_batch, y_hat)
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