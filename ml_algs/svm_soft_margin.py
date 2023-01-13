from scipy import optimize
import numpy as np

class MaxMarginClassifier:

  def __init__(self, C):
    self.alpha = None
    self.w = None
    self.supportVectors = None
    self.C = C

  def fit(self, X, y):
    N = len(y)

    # Gram matrix of (X,y)
    Xy = X*y[:,np.newaxis] # Not dot product because we scale x_i by y_i, see above formula
    GramXy =np.matmul(Xy,Xy.T) # This is equal to Xy.dot(Xy) or np.dot(Xy,XY)

    # Dual form of SVM, The Lagrangian Function
    def Ld0(G, alpha): 
      return alpha.sum() - 0.5*alpha.dot(alpha.dot(G))  # above eqn
    
    # Derivative of Lagrangian Function
    def partialDerivationLd0(G, alpha):
      return np.ones_like(alpha) - alpha.dot(G)
    
    # initialize alpha 0
    alpha = np.ones(N)

    # Solving for alpha using minimize tool (instead of SMO)
    A = np.vstack((-np.eye(N), np.eye(N)))
    b = np.concatenate((np.zeros(N), self.C * np.ones(N)))
    
    # b here is either C or 0
    # b = C for +ve alpha 
    # b = 0 for -ve alpha
    constraints = ({'type': 'eq', 'fun':lambda a: np.dot(a,y), 'jac': lambda a: y },
                   {'type': 'ineq', 'fun':lambda a: b - np.dot(A, a), 'jac': lambda a: -A}) 

    # Dual Lagrangian will maximize w.r.to alphas.
    # So, we pass -ve(Lagrangian) which will minimize w.r.to alphas.
    optRes = optimize.minimize(fun = lambda a: -Ld0(GramXy, a),
                               x0 = alpha,
                               method = 'SLSQP',
                               jac = lambda a: - partialDerivationLd0(GramXy, a),
                               constraints = constraints)
    self.alpha = optRes.x

    # now we got alphas
    # Next step is to find 'W'
    self.w =  np.sum(( self.alpha[:, np.newaxis] * Xy), axis = 0)

    epsilon = 1e-4
    self.supportVectors = X[ self.alpha > epsilon]
    self.supportLabels = y[self.alpha > epsilon]

    #Next step is to find 'b' (intercept)
    # self.intercept = self.supportLabels[0] - np.matmul(self.supportVectors[0].T, self.w)

    # original approach
    '''
    Issue: Seperator is not exactly at mid way b/w the margins
    '''
    self.b1 = self.supportLabels[0] - np.matmul(self.supportVectors[0].T, self.w)
    
    # 2nd approach: Averaging over all b estimates
    '''
    1. This approach is from Bishop book
    2. Seperator is exactly at mid way b/w the margins
    ''' 
    b = []
    for i in range(len(self.supportLabels)):
      b_i = self.supportLabels[i] - np.matmul(self.supportVectors[i].T, self.w)
      b.append( b_i )
    
    self.b2 = sum(b)/len(b)

    # self.intercept = self.b1  # original approach 
    self.intercept = self.b2  # modified approach

  def predict(self, X):
    
    return 2*(np.matmul(X, self.w) + self.intercept > 0) - 1
