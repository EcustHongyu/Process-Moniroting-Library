import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import KernelPCA


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        '''
        Dynamic Kernel Principle Components Analysis (DKPCA) for fault detection
        
        Map the vectors to a high-dimensional space using kernel function
        
        Parameters
        ----------
        self.cum_ratio : float, default=0.75
        Cumulative variance ratio of DKPCA
        
        self.kernel : str, kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'} or callable, default='rbf'
        Kernel used for DKPCA.

        self.gamma : float, default=None
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other kernels. If gamma is None, then it is set to 1/n_features.

        self.degree : int, default=3
        Degree for poly kernels. Ignored by other kernels

        self.time_lags : int, default=1
        Number of time lags to construct dynamic feature matrix.
        '''
        self.cum_ratio = configs.cum_ratio
        self.kernel = configs.kernel
        self.gamma = configs.gamma
        self.degree = configs.degree
        self.time_lags = configs.time_lags
        self.dkpca = None

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
        Fit the DKPCA model

        Parameters
        ----------
        train_data : array-like of shape [n_samples, n_features]
        self.n_components : the number of principle components
        self.dkpca : model
        lagged_train_data : array-like of shape [N-lags, D*(lags+1)]

        Returns
        -------
        None
        Different from PCA, class DKPCA do not have attribute 'explained_variance_ratio_'.
        Therefore, a manual computation of the explained variance ratio is required.
        '''
        lagged_train_data = self._construct_lagged_matrix(train_data)
        self.mean = np.mean(lagged_train_data, axis=0)
        lagged_train_data_centered = lagged_train_data - self.mean

        dkpca = KernelPCA(kernel=self.kernel, 
                     gamma=self.gamma, degree=self.degree, fit_inverse_transform=True)
        dkpca.fit(lagged_train_data_centered)

        total_var = np.sum(dkpca.eigenvalues_)
        Cumulative_CovRatio = np.cumsum(dkpca.eigenvalues_ / total_var)
        self.n_components = np.searchsorted(Cumulative_CovRatio, self.cum_ratio) + 1
        
        self.dkpca = KernelPCA(n_components=self.n_components, kernel=self.kernel, 
                     gamma=self.gamma, degree=self.degree, fit_inverse_transform=True)
        self.dkpca.fit(lagged_train_data_centered)

    def detect(self, test_data):
        '''
        Detect faults using DKPCA
        
        Parameters
        ----------
        test_data : array-like of shape [n_samples, n_features]
        lagged_test_data : array-like of shape [N-lags, D*(lags+1)]
        score : score matrix. The matrix after projection
        rec : reconstruction matrix. Reconstruct the score matrix back into the original feature space.
        t2 : Hotelling's T2 statistics. Compute the distance along each principle component direction.
            And normalize each direction by the variance matrix to eliminate the impact of variance discrepancies among features. 

        Returns
        -------
        res : residual of the reconstruction matrix. [n_samples-lags, ]
        t2 : Hotelling's T2 statistics. [n_samples-lags, ]
        '''
        lagged_test_data = self._construct_lagged_matrix(test_data)
        lagged_test_data_centered = lagged_test_data - self.mean

        score = self.dkpca.transform(lagged_test_data_centered)
        rec = self.dkpca.inverse_transform(score)
        res = np.mean((lagged_test_data - rec) ** 2, axis=1)

        t2 = np.sum((score ** 2) / self.dkpca.eigenvalues_, axis=1)

        return res, t2

    def forward(self, x):
        '''
        forward propagation interface
        '''
        res, t2 = self.detect(x)
        return res, t2