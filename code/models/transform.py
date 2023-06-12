from __future__ import print_function, division
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 01:44:09 2023

@author: Tingpeng Yang
"""
import einops
import numpy as np
import torch
import torch.nn as nn

class PositionalEncoder(torch.nn.Module):
    """The positional encoder for sequences.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    """

    def __init__(self, dim_model, max_wavelength=10000):
        """Initialize the MzEncoder"""
        super().__init__()

        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin
        scale = max_wavelength / (2 * np.pi)

        sin_term = scale ** (torch.arange(0, n_sin).float() / (n_sin - 1))
        cos_term = scale ** (torch.arange(0, n_cos).float() / (n_cos - 1))
        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode positions in a sequence.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_sequence, n_features)
            The first dimension should be the batch size and the second dimension should be the sequence

        Returns
        -------
        torch.Tensor of shape (batch_size, n_sequence, n_features)
        """
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], axis=2)
        return encoded + X

class EmbeddingTransform(nn.Module):

    def __init__(self, nin, nout, dropout=0.5, nhead=1,num_layers=6,activation=nn.ReLU()):
        super(EmbeddingTransform, self).__init__()
        self.nin = nin
        self.nout = nout
        self.dropout_p = dropout       
        self.embedding = nn.Embedding(5,nin,padding_idx=0)
        self.position_embedding=PositionalEncoder(nin)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=nin,
            nhead=nhead, 
            dim_feedforward=nin*2,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        ) 
        self.transform = nn.Linear(nin, nout)
        self.activation = activation
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        """
        :param x: Input RNAs (b,l,1)
        :type x: torch.Tensor
        :return: embedding (b,l,nout)
        :type: torch.Tensor
        """
        mask = ~x.sum(dim=2).bool()
        #print(mask)
        x = self.embedding(x).squeeze(2) #(b,l,d_model)
        x = self.position_embedding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.drop(self.activation(self.transform(x)))
        return x