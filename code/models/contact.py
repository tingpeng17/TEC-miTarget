# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 01:44:31 2023

@author: Tingpeng Yang
"""

import torch
import torch.nn as nn

class ContactCNN(nn.Module):

    def __init__(self, ks,projection_dim):
        super(ContactCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(2 * projection_dim, projection_dim, ks,padding=ks//2)
        self.batchnorm1 = nn.BatchNorm2d(projection_dim)
        self.activation1 = nn.ReLU()
        
        
        self.conv2 = nn.Conv2d(projection_dim, projection_dim//2, ks, padding=ks // 2)
        self.batchnorm2 = nn.BatchNorm2d(projection_dim//2)
        self.activation2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(projection_dim//2, projection_dim//4, ks, padding=ks // 2)
        self.batchnorm3 = nn.BatchNorm2d(projection_dim//4)
        self.activation3 = nn.ReLU()

        self.conv4 = nn.Conv2d(projection_dim//4, 1, ks, padding=ks // 2)
        self.batchnorm4 = nn.BatchNorm2d(1)
        self.activation4 = nn.Sigmoid()
        self.clip()

    def clip(self):
        # Force the convolutional layer to be transpose invariant.
        self.conv2.weight.data[:] = 0.5 * (self.conv2.weight + self.conv2.weight.transpose(2, 3))

    def forward(self, z0, z1):
        
        # z0 is (b,N,d), z1 is (b,M,d)
        z0 = z0.transpose(1, 2)
        z1 = z1.transpose(1, 2)
        # z0 is (b,d,N), z1 is (b,d,M)

        z_dif = torch.abs(z0.unsqueeze(3) - z1.unsqueeze(2))
        z_mul = z0.unsqueeze(3) * z1.unsqueeze(2)
        z_cat = torch.cat([z_dif, z_mul], 1)
        
        x = self.conv1(z_cat)
        x = self.activation1(x)
        x = self.batchnorm1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        
        x = self.conv3(x)
        x = self.activation3(x)
        x = self.batchnorm3(x)
        
        x = self.conv4(x)
        x = self.activation4(x)
        x = self.batchnorm4(x)
        
        return x