import numpy as np

class SVM():

    def __init__(self, learningRate = 0.0001, lamda = 0.001, nIters=1000):

        self.learningRate = learningRate
        self.lamda = lamda
        self.nIters = nIters
    
    def fit(self, X, y):
        
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
    
    def predict(self, X):
        pred_ = np.dot(self.w, X) + self.b
        return np.sign(pred_)