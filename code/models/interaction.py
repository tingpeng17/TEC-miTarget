# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 01:45:00 2023

@author: Tingpeng Yang
"""

import numpy as np
import torch
import torch.nn as nn

class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:

    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`

    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \\geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, x0=0, k=1, train=False):
        super(LogisticActivation, self).__init__()
        self.x0 = x0
        if train:
            self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        else:
            self.k = k

    def forward(self, x):
        x = torch.clamp(1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1).squeeze()
        return x
        
class ModelInteraction(nn.Module):
    def __init__(
        self,
        args,
        embedding_transform,
        contact,
        do_w=True,
        do_sigmoid=True,
        do_pool=False,
        pool_size=9,
        theta_init=1,
        lambda_init=0,
        gamma_init=0,
        p0=0.5
    ):
        super(ModelInteraction, self).__init__()
        self.device=args.device
        self.do_w = do_w
        self.do_sigmoid = do_sigmoid
        if do_sigmoid:
            self.activation = LogisticActivation(x0=p0, k=20, train=True)
        self.embedding_transform = embedding_transform
        self.contact = contact
        if self.do_w:
            self.theta = nn.Parameter(torch.FloatTensor([theta_init]))
            self.lambda_ = nn.Parameter(torch.FloatTensor([lambda_init]))
        self.do_pool = do_pool
        self.maxPool = nn.MaxPool2d(pool_size, padding=pool_size // 2)
        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))
        self.clip()

    def clip(self):
        # Clamp model values
        self.contact.clip()
        if self.do_w:
            self.theta.data.clamp_(min=0, max=1)
            self.lambda_.data.clamp_(min=0)
        self.gamma.data.clamp_(min=0)

    def map_predict(self, z0, z1):
        if self.embedding_transform is not None:
            z0=self.embedding_transform(z0)
            z1=self.embedding_transform(z1)
        C = self.contact.forward(z0, z1)
        if self.do_w:
            # Create contact weighting matrix
            N, M = C.shape[2:]
            x1 = torch.from_numpy(-1 * ((np.arange(N) + 1 - ((N + 1) / 2)) / (-1 * ((N + 1) / 2)))** 2).float()
            x1 = x1.to(self.device)
            # x1 = torch.exp(self.lambda1 * x1)
            x1 = torch.exp(self.lambda_ * x1)
            x2 = torch.from_numpy(-1 * ((np.arange(M) + 1 - ((M + 1) / 2)) / (-1 * ((M + 1) / 2)))** 2).float()
            x2 = x2.to(self.device)
            # x2 = torch.exp(self.lambda2 * x2)
            x2 = torch.exp(self.lambda_ * x2)
            W = x1.unsqueeze(1) * x2
            W = (1 - self.theta) * W + self.theta
            yhat = C * W
        else:
            yhat = C
        if self.do_pool:
            yhat = self.maxPool(yhat)
        # Mean of contact predictions where p_ij > mu + gamma*sigma
        mu = torch.mean(yhat,dim=(1,2,3)).repeat(yhat.shape[2]*yhat.shape[3],1).T.reshape(yhat.shape[0],yhat.shape[1],yhat.shape[2],yhat.shape[3])
        sigma = torch.var(yhat,dim=(1,2,3)).repeat(yhat.shape[2]*yhat.shape[3],1).T.reshape(yhat.shape[0],yhat.shape[1],yhat.shape[2],yhat.shape[3])
        Q = torch.relu(yhat - mu - (self.gamma * sigma))
        phat = torch.sum(Q,dim=(1,2,3)) / (torch.sum(torch.sign(Q),dim=(1,2,3)) + 1)
        if self.do_sigmoid:
            phat = self.activation(phat)
        return C, phat