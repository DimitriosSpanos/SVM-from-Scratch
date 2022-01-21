"""
Neural Networks - Deep Learning
Heart Disease Predictor ( Binary Classification )
Author: Dimitrios Spanos Email: dimitrioss@ece.auth.gr
"""
import numpy as np
from cvxopt import matrix, solvers

# ------------
#   Kernels
# ------------

def poly(x, z, d=3, coef=1, g=1):
    return (g * np.dot(x, z.T) + coef) ** d

def rbf(x, z, sigma):
    return np.exp(-np.linalg.norm(x-z,axis=1)**2 / (2*(sigma**2)))

def linear(x, z):
    return np.matmul(x, z.T)

def sigmoid(x, z, g=1, coef=0):
    return np.tanh(g * np.dot(x, z.T) + coef)

# ------------
#     SVM
# ------------

class my_SVM:

    def __init__(self, C, kernel='linear', sigma=1):
        self.C = C
        self.kernel = kernel
        self.sigma = sigma
        self.sv = 0
        self.sv_y = 0
        self.alphas = 0
        self.w = 0
        self.b = 0

    def fit(self, X, y):

        # Calculate the Kernel(xi,xj)
        m, n = X.shape
        K = np.zeros((m,m))
        if self.kernel == 'rbf':
            for i in range(m):
                K[i,:] = rbf(X[i,np.newaxis], X,  sigma=self.sigma)
        elif self.kernel == 'poly':
            for i in range(m):
                K[i,:] = poly(X[i,np.newaxis], X)
        elif self.kernel == 'sigmoid':
            for i in range(m):
                K[i,:] = sigmoid(X[i,np.newaxis], X)
        elif self.kernel == 'linear':
            for i in range(m):
                K[i,:] = linear(X[i,np.newaxis], X)


        # Solve the QP Problem
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones((m, 1)))
        A = matrix(matrix(y.T), (1, m), 'd')
        b = matrix(np.zeros(1))
        G = matrix(np.vstack((np.eye(m)*-1, np.eye(m))))
        h = matrix(np.hstack((np.zeros(m),np.ones(m)*self.C)))
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)

        # Get the solution's results
        alphas = np.array(solution['x'])
        S = (alphas > 1e-4).flatten()
        self.sv = X[S]
        self.sv_y = y[S]
        self.w = np.dot((y.reshape(-1,1) * alphas).T, X)[0]
        self.alphas = alphas[S] # get rid of alphas ~= 0
        self.b = np.mean(self.sv_y - np.dot(self.sv, self.w.T))

        #print("w:", self.w)
        #print("b:", self.b)


    def predict(self, X):

        K_xi_x = 0
        if self.kernel == 'rbf':
            K_xi_x = rbf(self.sv, X, self.sigma)
        elif self.kernel == 'poly':
            K_xi_x = poly(self.sv, X)
        elif self.kernel == 'sigmoid':
            K_xi_x = sigmoid(self.sv, X)
        elif self.kernel == 'linear':
            K_xi_x = linear(self.sv, X)

        sum = 0
        for i in range(len(K_xi_x)):
            sum +=self.alphas[i] * self.sv_y[i]* K_xi_x[i]

        prod = sum + self.b
        prediction = np.sign(prod)
        return prediction