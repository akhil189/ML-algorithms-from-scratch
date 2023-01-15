import numpy as np

class NaiveBayes():
    """Implements Naive Bayes Algorithm for continuous data.
    
    Assumption: The data follows a normal distribution.
    """

    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Learns the mean, variance and priors for each class.
        
        X: The data matrix, shape = (n_samples, n_features).

        y: The class labels, shape = (n_samples,).
        """
        
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y) # sorted class labels
        self.n_classes = len(self.classes_) # number of classes

        self.mean = np.zeros((self.n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((self.n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(self.n_classes, dtype=np.float64)

        for idx, class_ in enumerate(self.classes_):
            X_class = X[y == class_]
            self.mean[idx, :] = X_class.mean(axis=0)
            self.var[idx, :] = X_class.var(axis=0)
            self.priors[idx] = X_class.shape[0] / float(n_samples)

    def pdf(self, class_idx: int, x: np.ndarray) -> np.ndarray:
        """Computes the Gaussian probability density function for each feature.
        
        Args:
            class_idx (int): Index of the class.

            x: A sample feature vector.
        
        Returns:
            pdf: Probability density function values for each feature.
        """

        mean = self.mean[class_idx, :]
        var = self.var[class_idx, :]
        pdf = 1/(np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2/ (2*var))

        return pdf

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the class labels for the provided data.
        
        Args:
            X: The data matrix, shape = (n_samples, n_features).
        
        Returns:
            y_hat: The predicted class labels.
        """

        n_samples, n_features = X.shape
        
        y_hat = []
        for x in X:
            posteriors = []
            for idx, class_ in enumerate(self.classes_):
                log_prior = np.log(self.priors[idx])
                log_class_conditional = np.sum(np.log(self.pdf(idx, x)))
                log_posterior = log_class_conditional + log_prior
                posteriors.append(log_posterior)
            
            y_hat.append(self.classes_[np.argmax(posteriors)])
        
        return np.array(y_hat)