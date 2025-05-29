import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        '''
        Dynamic Principle Components Analysis (DPCA) for fault detection
        
        Map the vectors to a high-dimensional space using kernel function
        
        Parameters
        ----------
        self.cum_ratio : float, default=0.75
        Cumulative variance ratio of DPCA
        
        self.time_lags : int, default=1
        Number of time lags to construct dynamic feature matrix.
        '''
        self.cum_ratio = configs.cum_ratio
        self.time_lags = configs.time_lags
        self.dpca = None

    def _construct_lagged_matrix(self, data):
        '''
        Construct a feature matrix incorporating time lags

        Parameters
        ----------
        data : array-like of shape [n_samples, n_features]

        Returns
        -------
        lagged_matrix : array-like of shape [n_samples-self.time_lags, n_features*(self.time_lags+1)]
        '''
        n_samples, n_features = data.shape
        lagged_data = []
        for i in range(self.time_lags + 1):
            lagged_data.append(data[i:n_samples-self.time_lags+i, :])
        return np.concatenate(lagged_data, axis=1)

    def fit(self, train_data):
        '''
        Fit the DPCA model
        
        Parameters
        ----------
        train_data : array-like of shape [n_samples, n_features]
        self.n_components : the number of principle components
        self.dpca : model
        lagged_train_data : array-like of shape [N-lags, D*(lags+1)]

        Returns
        -------
        None
        '''
        lagged_train_data = self._construct_lagged_matrix(train_data)
        self.mean = np.mean(lagged_train_data, axis=0)
        lagged_train_data_centered = lagged_train_data - self.mean
        
        dpca = PCA()
        dpca.fit(lagged_train_data_centered)
        
        Cumulative_CovRatio = np.cumsum(dpca.explained_variance_ratio_)
        self.n_components = np.searchsorted(Cumulative_CovRatio, self.cum_ratio) + 1
        
        self.dpca = PCA(n_components=self.n_components)
        self.dpca.fit(lagged_train_data_centered)

    def detect(self, test_data):
        '''
        Detect faults using DPCA
        
        Parameters
        ----------
        test_data : array_like of shape [n_samples, n_features]
        lagged_test_data : array-like of shape [N-lags, D*(lags+1)]
        score : score matrix. The matrix after projection
        rec : reconstruction matrix. Reconstruct the score matrix back into the original feature space.
        t2 : Hotelling's T2 statistics. Compute the distance along each principle component direction.
            And normalize each direction by the variance matrix to eliminate the impact of variance discrepancies among features. 

        Returns
        -------
        res : residual of the reconstruction matrix. [N-lags, ]
        t2 : Hotelling's T2 statistics. [N-lags, ]
        '''
        lagged_test_data = self._construct_lagged_matrix(test_data)
        lagged_test_data_centered = lagged_test_data - self.mean

        score = self.dpca.transform(lagged_test_data_centered)
        rec = self.dpca.inverse_transform(score)
        res = np.mean((lagged_test_data_centered - rec) ** 2, axis=1)

        t2 = np.sum((score ** 2) / self.dpca.explained_variance_, axis=1)

        return res, t2

    def forward(self, x):
        '''
        forward propagation interface
        '''
        res, t2 = self.detect(x)
        return res, t2