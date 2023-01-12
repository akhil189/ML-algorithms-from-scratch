import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

class LogisticRegression:

    def __init__(self, learningRate, tolerance, maxIteration = 5000, outliers = None):

        self.learningRate = learningRate
        self.tolerance = tolerance
        self.maxIteration = maxIteration
        self.outliers = outliers
  
    def datasetReader(self):
        train_df = pd.read_excel('./datasets/data.xls', sheet_name='2004--2005 Data')
        test_df = pd.read_excel('./datasets/data.xls', sheet_name='2004--2007 Data')
        train_df, test_df = np.array(train_df, dtype=np.float64), np.array(test_df, dtype=np.float64)

        X_train, y_train = train_df[:, 1:], train_df[:, 0]
        X_test, y_test = test_df[:, 1:], test_df[:, 0]

        return X_train, y_train, X_test, y_test

    def addX0(self, X):
        return np.column_stack([np.ones([X.shape[0],1]), X])
    
    def sigmoid(self, z):
        sig = 1/( 1+np.exp(-z) )
        return sig

    def costFunction(self, X, y):

        #approach 1
        #pred_ = np.log(np.ones(X.shape[0]) + np.exp(X.dot(self.w))) - X.dot(self.w).dot(y) # Negative Log-likelihood
        pred_ = np.log(np.ones(X.shape[0]) + np.exp(X.dot(self.w))) - X.dot(self.w)*y
        cost = pred_.sum()

        # approach 2
        # sig = self.sigmoid(X.dot(self.w))
        # pred_ = y * np.log(sig) + (1-y) * np.log(1-sig)
        # cost = pred_.sum()

        # To DO: check which one is better.(speed test)

        return cost

    def gradient(self, X,y):
        sig = self.sigmoid(X.dot(self.w))
        grad = (sig - y).dot(X)
        return grad
  
    def gradientDescent(self, X, y):
        costSequence = []
        lastCost = float('inf')

        for i in tqdm(range(self.maxIteration)):
            self.w = self.w - self.learningRate * self.gradient(X, y)
            currentCost = self.costFunction(X, y)
            diff = lastCost - currentCost
            lastCost = currentCost
            costSequence.append(currentCost)

            # can use abs(diff)
            if abs(diff) < self.tolerance:
                print("The model stopped - No Further Improvment")
                break
        self.plotCost(costSequence)

        return
  
    def plotCost(self, costSequence):
        s = np.array(costSequence)
        t = np.arange(s.size)

        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(xlabel = 'iterations', ylabel = 'cost', title = 'cost trend')
        ax.grid()

        plt.legend(bbox_to_anchor = (1.05, 1), shadow=True)
        plt.show()

    def predict(self, X):
        sig = self.sigmoid(X.dot(self.w)) # X.dot(w) is same as w.T.dot(X)    

        return np.around(sig)

    def evaluate(self, y, y_hat):
        y = (y == 1) # same as y.sum()
        y_hat = (y_hat == 1)

        accuracy = (y == y_hat).sum() /y.size
        precision = (y & y_hat).sum() /y_hat.sum()
        recall = (y & y_hat).sum() / y.sum()

        return accuracy, precision, recall

    def runModel(self):
        self.X_train, self.y_train, self.X_test, self.y_test = self.datasetReader()
        if self.outliers:
            self.remove_index(self.outliers)
        
        self.w = np.ones(self.X_train.shape[1], dtype= np.float64) * 0
        
        print(f'X_train:{self.X_train.shape}, y_train:{self.y_train.shape}')
        print(f'X_test:{self.X_test.shape}, y_test:{self.y_test.shape}')
        self.gradientDescent(self.X_train, self.y_train)
        
        # print(self.X_train)
        # print(self.y_train)
        # print(self.w)
        
        y_hat_train = self.predict(self.X_train)
        accuracy, recall, precision = self.evaluate(self.y_train, y_hat_train)

        print(f'Training Accuracy: {accuracy}')
        print(f'Training Precision: {precision}')
        print(f'Training Recall: {recall}')

        y_hat_test = self.predict(self.X_test)
        accuracy, recall, precision = self.evaluate(self.y_test, y_hat_test)

        print(f'Testing Accuracy: {accuracy}')
        print(f'Testing Precision: {precision}')
        print(f'Testing Recall: {recall}')

    def remove_index(self, outliers):
        self.X_train = np.delete(self.X_train, outliers, axis=0)
        self.y_train = np.delete(self.y_train, outliers, axis=0)

        #self.X_test = np.delete(self.X_test, outliers, axis=0)
        #self.y_test = np.delete(self.y_test, outliers, axis=0)

    def plot(self):
        
        fig = plt.figure(figsize = (12, 8))
        ax1 = fig.add_subplot(1,2,1, projection = '3d')
        ax2 = fig.add_subplot(1,2,2, projection = '3d')
        
        # Data for three-dimensional scattered points
        ax1.scatter3D(self.X_train[:, 0], self.X_train[:, 1], 
                     self.sigmoid(self.X_train.dot(self.w)), 
                     c = self.y_train[:], cmap='viridis', s=100);
        ax1.set_title('X_train')
        ax1.set_xlim3d(55, 80)
        ax1.set_ylim3d(80, 240)
        ax1.set_xlabel('$x_1$ feature', fontsize=15)
        ax1.set_ylabel('$x_2$ feature', fontsize=15, )
        ax1.set_zlabel('$P(Y = 1|x_1, x_2)$', fontsize=15, rotation = 0)    

        # Plotting for test set
        # Data for three-dimensional scattered points
        ax2.scatter3D(self.X_test[:, 0], self.X_test[:, 1], 
                     self.sigmoid(self.X_test.dot(self.w)), 
                     c = self.y_test[:], cmap='viridis', s=100);
        ax2.set_title('X_test')
        ax2.set_xlim3d(55, 80)
        ax2.set_ylim3d(80, 240)
        ax2.set_xlabel('$x_1$ feature', fontsize=15)
        ax2.set_ylabel('$x_2$ feature', fontsize=15, )
        ax2.set_zlabel('$P(Y = 1|x_1, x_2)$', fontsize=15, rotation = 0)    
        plt.show()

    def scatterPlt(self):
        
        # evenly sampled points
        x_min, x_max = 55, 80
        y_min, y_max = 80, 240

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                              np.linspace(y_min, y_max, 250))
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = grid.dot(self.w).reshape(xx.shape)
        
        #f, ax = plt.subplots(figsize=(14,12))
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,10))

        ax1.contour(xx, yy, probs, levels=[0.5], cmap="Greys", vmin=0, vmax=.6)
        ax1.scatter(self.X_train[:, 0], self.X_train[:, 1], 
                    c=self.y_train[:], s=50,
                    cmap="RdBu", vmin=-.2, vmax=1.2,
                    edgecolor="white", linewidth=1)
        
        ax1.set(xlabel = 'x1 feature', ylabel = 'x2 feature')
        ax1.set_title('X_train')
        
        # Plotting for Test
        ax2.contour(xx, yy, probs, levels=[0.5], cmap="Greys", vmin=0, vmax=.6)
        ax2.scatter(self.X_test[:, 0], self.X_test[:, 1], 
                    c=self.y_test[:], s=50,
                    cmap="RdBu", vmin=-.2, vmax=1.2,
                    edgecolor="white", linewidth=1)

        ax2.set(xlabel = 'x1 feature', ylabel = 'x2 feature')
        ax2.set_title('X_test')
        plt.show()
        
        
    def plot3D(self):
        # evenly sampled points
        x_min, x_max = 55, 80
        y_min, y_max = 80, 240

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                             np.linspace(y_min, y_max, 250))

        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = grid.dot(self.w).reshape(xx.shape)
        fig = plt.figure(figsize=(14,12))
        ax = plt.axes(projection='3d')
        ax.contour3D(xx, yy, probs, 50, cmap='binary')

        ax.scatter3D(self.X_train[:, 0], self.X_train[:, 1], 
                    c=self.y_train[:], s=50,
                    cmap="RdBu", vmin=-.2, vmax=1.2,
                    edgecolor="white", linewidth=1)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('probs')
        ax.set_title('3D contour')
        plt.show()