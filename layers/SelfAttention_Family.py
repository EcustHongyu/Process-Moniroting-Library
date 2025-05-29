import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, 
                 scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # Queries: [B, L, H, E], i.e. 
        # [Batch_size, Query sequence length, Number of attention heads, Dimension of embedding] 
        # Keys: [B, S, H, E], i.e.
        # [Batch_size, Key sequence length, Number of attention heads, Dimension of embedding]
        # Values: [B, S, H, E], i.e.
        # [Batch_size, Value sequence length, Number of attention heads, Dimension of embedding]
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # If the scaling parameter is specified, then use the specified parameters.
        # Else use the default parameter in the paper: 1/sqrt(E)
        scale = self.scale or 1. / sqrt(E)

        # simplify the calculation of matrix multiplication by using einstein summation convention 
        # that means there is no need to transpose the matrix when calculating
        # Q * K.T: (Original:[B, L, H, E])Transpose:[B, H, L, E] * 
        # (Original:[B, S, H, E])Transpose:[B, H, E, S] -> [B, H, L, S]
        scores = torch.einsum('blhe, bshe->bhls', queries, keys)

        if self.mask_flag:
            # if attn_mask matrix is not specified, assign a boolen type 
            # upper triangular matrix to it.
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            # assign negative infinity to the upper triangular part of the matrix
            # since in time series or nlp tasks, the future information cannot be used. 
            # only past and cuurent information is available.
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('bhls, bshe->bhle', A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
        
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries = queries,
            keys = keys,
            values = values,
            attn_mask = attn_mask,
            tau = tau,
            delta = delta
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), attn


