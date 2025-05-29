import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        '''
        Slow Feature Analysis (SFA) for fault detection
        
        Minimize the temporal fluctuation of the projected features
        
        Parameters
        ----------
        self.ratio_sfa : float, default=0.2, the ratio of slow features in SFA
        self.input_dim : the dimension of input features to determine the parameters in mlp
        '''
        self.ratio_sfa = configs.ratio_sfa
        self.input_dim = configs.input_dim
        self.n_components = int(self.input_dim*self.ratio_sfa)
        self.sfa = nn.Linear(self.input_dim, self.n_components, bias=False)
        if torch.cuda.is_available() and configs.use_gpu:
            self.device = torch.device(f'cuda:{configs.gpu}')
            print(f'Using GPU: {configs.gpu}')
        else:
            self.device = torch.device('cpu')
            print(f'Using CPU')

    def fit(self, train_data):
        '''
        Fit the SFA model
        
        Parameters
        ----------
        train_data : array-like of shape [n_samples, n_features]
        self.n_components : the number of principle components
        self.sfa : model

        Returns
        -------
        None

        A_diff : first-order differences are employed as a discrete approximation
               of the temporal derivative of the features. [N-1, D]
        cov_diff : the covariance matrix of the difference matrix. [D, D]
        eigvals : the eigen values of the matrix cov_diff. [D,]
        eigvecs : the eigen vectors of the matrix cov_diff. [D, D]
        
        '''
        pca = PCA(whiten=True)
        train_data = pca.fit_transform(train_data)

        A_diff = train_data[1:] - train_data[:-1]
        A_diff = torch.tensor(A_diff, dtype=torch.float32).to(self.device)
        cov_diff = torch.matmul(A_diff.T, A_diff) / (A_diff.shape[0] - 1)
        
        cov_B = train_data.T @ train_data
        cov_B = torch.tensor(cov_B, dtype=torch.float32).to(self.device)
        
        eigvals, eigvecs = torch.linalg.eigh(torch.linalg.inv(cov_B) @ cov_diff) 
        
        idx = torch.argsort(eigvals)
        self.eigvecs = eigvecs[:, idx]
        self.eigvals = eigvals[idx]

        # use .data attribute to directly assign a tensor to the weight. 
        # (Or only nn.parameter or None can be assigned)
        # self.sfa.weight.data = self.eigvecs[:, :self.n_components].T
        self.sfa.weight.data = self.eigvecs[:, -self.n_components:].T
        self.sfa.weight.requires_grad = False

    def detect(self, test_data):
        '''
        Detect faults using SFA
        
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
        test_data = torch.tensor(test_data, dtype=torch.float32).to(self.device)
        score = self.sfa(test_data)
        rec = score @ self.sfa.weight
        res = torch.mean((test_data - rec) ** 2, axis=1)
        
        t2 = torch.sum((score ** 2), axis=1)

        return res, t2

    def forward(self, x):
        '''
        forward propagation interface
        '''
        res, t2 = self.detect(x)
        return res.detach().cpu().numpy(), t2.detach().cpu().numpy()