import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        '''
        Principle Components Analysis (PCA) for fault detection
        
        Maximum the variance after projection
        X : array-like of shape [n_samples, n_features]
        W : projection matrix [n_features, n_reduced_features]
        '*' below represents matrix multiplication
        the input X is zero-meaning normalization to ensure X*X_t is the covariance matrix
        
        Max L(W) = W_t * X_t * X * W
        s.t. W * W_t = I
        
        Let S=X_t*X(i.e. covariance matrix), use lagrangian multiplier, the constrained extremum problem
        can be transformed into an unconstrained extremum problem:
        Max L(W,lambda) = W_t * X_t * X * W + lambda(W * W_t - I)
        
        Calculate the partial derivative of L with respect to W, then the original optimization problem
        can be transformed into:
        X_t * X * W = W * Lambda (Lambda is a matrix different from scalar lambda),
        which is a typical eigen vector decomposition problem.

        Perform svd on matrix X, then we get:
        X = U * Sigma * V_t, 
        where U is the left singular matrix,
        Sigma is a diagonal matrix representing singular value,
        V_t is the right singular matrix.
        X_t * X = V * Sigma * U_t * U * Sigma * V_t = V * Sigma**2 *V_t.
        Hence, V is the projection matrix, Sigma**2 is the eigen value matrix (diagonal matrix)
        For a new X, X_new = X * V[:, :n_components]
                           = U[:, :n_components] * Sigma[:n_components, :n_components] * V_t[:n_components, :] * V[:, :n_components] 
                           = U[:, :n_components] * Sigma[:n_components, :n_components]

        Parameters
        ----------
        Only one parameter is accepted by PCA.
        self.cum_ratio: cumulative variance ratio of PCA
        '''
        self.cum_ratio = configs.cum_ratio
        self.pca = None

    def fit(self, train_data):
        '''
        Fit the PCA model
        
        Parameters
        ----------
        train_data : array-like of shape [n_samples, n_features]
        self.n_components : the number of principle components
        self.pca : model

        Returns
        -------
        None

        Determine the number of self.n_components.
        Then train self.pca using self.n_components.
        '''
        pca = PCA()
        pca.fit(train_data)
        Cumulative_CovRatio = np.cumsum(pca.explained_variance_ratio_)
        self.n_components = np.searchsorted(Cumulative_CovRatio, self.cum_ratio) + 1
        
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(train_data)
        
    def detect(self, test_data):
        '''
        Detect faults using PCA
        
        Parameters
        ----------
        test_data : array_like of shape [n_samples, n_features]
        score : score matrix. The matrix after projection
        rec : reconstruction matrix. Reconstruct the score matrix back into the original feature space.
        t2 : Hotelling's T2 statistics. Compute the distance along each principle component direction.
            And normalize each direction by the variance matrix to eliminate the impact of variance discrepancies among features. 

        Returns
        -------
        res : residual of the reconstruction matrix. [n_samples,]
        t2 : Hotelling's T2 statistics. [n_samples, ]
        '''
        score = self.pca.transform(test_data)
        rec = self.pca.inverse_transform(score)
        res = np.mean((test_data - rec) ** 2, axis=1)

        t2 = np.sum((score ** 2) / self.pca.explained_variance_, axis=1)

        return res, t2

    def forward(self, x):
        '''
        forward propagation interface
        '''
        res, t2 = self.detect(x)
        return res, t2