import numpy as np

class PCA:
    """Principal Component Analysis(PCA)
    
    Attributes:
    ----------
        components: np.ndarray, The computed weights/loadings(eigen vectors of Covariance Matrix(C)).
            - Each column is an eigen vector of C in sorted order.

        explained_variance: The amount of variance explained by each of the selected components.
            - The eigen values in descending order.
        
        explained_variance_ratio: Percentage of variance explained by each of the selected components.

        cum_explained_variance: The cumulative percentage of variance explained by each of the selected components.
    """
    
    def __init__(self, n_components: int=None, scaling: str='mean_centering'):
        """
        Args:
            n_components: int, default=None
                Number of components to keep. 
                
                if n_components is not set all components are kept.

            sclaing: str, {'mean_centering', 'standardize'}, default='mean_centering
                
                'mean-centering': Use when all features(columns) are in the same scale(units)
                'standardize': Use when all features(columns) are in different scales(units)
        """
        
        self.n_components = n_components
        self.scaling = scaling
        
    def fit(self, X: np.ndarray):
        """
        It computes the matrix A -> weights/loadings(eigen vectors of C)
        -----
        Args:
            X: np.ndarray, The Data Matrix either standardized/mean-centered

            C: np.ndarray, C is computed as 1/n*(X^T*X)
               C is Covariance matrix if X is mean-centered 
               C is Correlation Coefficient matrix if X is standarized           

        Returns: None
        """

        #scaling
        if self.scaling == 'mean_centering':
            self.mean = np.mean(X, axis=0)
            X = X - self.mean
        elif self.scaling == 'standardize':
            self.std = np.std(X, axis=0)
            X = (X - self.mean)/self.std

        self.C = np.dot(X.T, X)/(np.shape(X)[0]-1) 

        '''alternate ways to compute covariance matrix.'''
        # cov = np.cov(X.T, bias=True) #(biased=population covariance, unbiased=sample covairance)
        # C = np.matmul(X.T, X)/(np.shape(X)[0]-1) can also be used

        # computing eigen values and eigen Vectors of C
        eigen_values, eigen_vectors = np.linalg.eig(self.C)

        # transposing eigen vectors into rows
        eigen_vectors = eigen_vectors.T 

        # finding indices in descening order of eigen values 
        ids = np.argsort(np.abs(eigen_values))[::-1]

        # rearranging the eigen values and eigen vectors in descending order
        self.eigen_values = eigen_values[ids]
        self.eigen_vectors = eigen_vectors[ids][:self.n_components] if self.n_components else eigen_vectors[ids]

        self.components = self.eigen_vectors.T
        self.explained_variance = self.eigen_values
        self.explained_variance_ratio = self._compute_explained_variance_ratio(self.eigen_values) 
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)
        
    def transform(self, X: np.ndarray):
        """
        Args:
            X: np.ndarray, The Data Matrix on which the transformation is to be applied i.e.,
               projected onto the components/loadings/weights(eigen vectors), computed earlier using PCA.fit()   
        
        Returns: The Transformed matrix, Z
        """
        
        #scaling
        if self.scaling == 'mean_centering':
            self.mean = np.mean(X, axis=0)
            X = X - self.mean
        elif self.scaling == 'standardize':
            self.std = np.std(X, axis=0)
            X = (X - self.mean)/self.std
        
        # Apply dimensionality reduction to X
        Z = np.dot(X, self.components) # //lar to np.matmul(X, self.components)
        
        return Z   
        
    def fit_transform(self, X: np.ndarray):
        """
        X: np.ndarray, The Data Matrix which is to be transformed i.e., on to which dimensionality reduction is applied.
           
            The weights/loadings are learnt using the same Data Matrix and the same Data Matrix is transformed using them.

        returns: The Transformed matrix, Z 
        """
        
        self.fit(X)
        Z = self.transform(X)
        
        return Z
    
    def _compute_explained_variance_ratio(self, eigen_values: np.ndarray):
        """Computes percentage of variance explained by each of the selected components.
        
        Args:
            eigen_values: np.ndarray
        """
        
        explained_variance_ratio = []
        total_variance = np.sum(eigen_values)
        return eigen_values/total_variance