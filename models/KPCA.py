import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import KernelPCA


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        '''
        Kernel Principle Components Analysis (KPCA) for fault detection
        
        Map the vectors to a high-dimensional space using kernel function
        
        Parameters
        ----------
        self.cum_ratio : float, default=0.75
        Cumulative variance ratio of KPCA
        
        self.kernel : str, kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'} or callable, default='rbf'
        Kernel used for KPCA.

        self.gamma : float, default=None
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other kernels. If gamma is None, then it is set to 1/n_features.

        self.degree : int, default=3
        Degree for poly kernels. Ignored by other kernels
        '''
        self.cum_ratio = configs.cum_ratio
        self.kernel = configs.kernel
        self.gamma = configs.gamma
        self.degree = configs.degree
        self.kpca = None

    def fit(self, train_data):
        '''
        Fit the KPCA model

        Parameters
        ----------
        train_data : array-like of shape [n_samples, n_features]
        self.n_components : the number of principle components
        self.kpca : model

        Returns
        -------
        None
        Different from PCA, class KPCA do not have attribute 'explained_variance_ratio_'.
        Therefore, a manual computation of the explained variance ratio is required.
        '''
        kpca = KernelPCA(kernel=self.kernel, 
                     gamma=self.gamma, degree=self.degree, fit_inverse_transform=True)
        kpca.fit(train_data)
        total_var = np.sum(kpca.eigenvalues_)
        Cumulative_CovRatio = np.cumsum(kpca.eigenvalues_ / total_var)
        self.n_components = np.searchsorted(Cumulative_CovRatio, self.cum_ratio) + 1
        
        self.kpca = KernelPCA(n_components=self.n_components, kernel=self.kernel, 
                     gamma=self.gamma, degree=self.degree, fit_inverse_transform=True)
        self.kpca.fit(train_data)

    def detect(self, test_data):
        '''
        Detect faults using KPCA
        
        Parameters
        ----------
        test_data : array-like of shape [n_samples, n_features]
        score : score matrix. The matrix after projection
        rec : reconstruction matrix. Reconstruct the score matrix back into the original feature space.
        t2 : Hotelling's T2 statistics. Compute the distance along each principle component direction.
            And normalize each direction by the variance matrix to eliminate the impact of variance discrepancies among features. 

        Returns
        -------
        res : residual of the reconstruction matrix. [n_samples, ]
        t2 : Hotelling's T2 statistics. [n_samples, ]
        '''
        score = self.kpca.transform(test_data)
        rec = self.kpca.inverse_transform(score)
        res = np.mean((test_data - rec) ** 2, axis=1)

        t2 = np.sum((score ** 2) / self.kpca.eigenvalues_, axis=1)

        return res, t2

    def forward(self, x):
        '''
        forward propagation interface
        '''
        res, t2 = self.detect(x)
        return res, t2