import torch
import torch.nn as nn
import numpy as np



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU() if configs.activation == 'relu' else nn.GELU(),
            nn.Linear(configs.d_model, configs.dec_in)
        )
        self.decoder = nn.Sequential(
            nn.Linear(configs.dec_in, configs.d_model),
            nn.ReLU() if configs.activation == 'relu' else nn.GELU(),
            nn.Linear(configs.d_model, 1)
        )
        # self.koopman_operator = nn.Parameter(torch.randn(configs.dec_in, configs.dec_in))
        K_init = torch.randn(configs.dec_in, configs.dec_in)
        U, _, V = torch.linalg.svd(K_init)
        self.koopman_operator = nn.Linear(configs.dec_in, configs.dec_in, bias=False)
        self.koopman_operator.weight.data = torch.mm(U, V.t())

    def forward(self, x, y):
        mean_x = x.mean(1, keepdim=True).detach() # B x 1 x E
        x = x - mean_x
        std_x = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std_x

        mean_y = y.mean(1, keepdim=True).detach() # B x 1 x E
        y = y - mean_y
        std_y = torch.sqrt(torch.var(y, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        y = y / std_y

        x = x.transpose(1, 2) # [B,L,D]->[B,D,L]
        y = y.transpose(1, 2) # [B,L,D]->[B,D,L]
        z = self.encoder(x) # [B,D,E]
        # z_pred = torch.matmul(z, self.koopman_operator) # [B,D,E]*[E,E]=[B,D,E]
        z_pred = self.koopman_operator(z)
        x_rec = self.decoder(z_pred) # [B,D,L]
        z_actual = self.encoder(y) # [B,D,E]

        x_rec = x_rec.transpose(1, 2)
        x_rec = x_rec * std_x + mean_x
        return x_rec, z_pred, z_actual
