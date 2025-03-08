import torch
import torch.nn as nn
import torch.nn.functional as F
from .discriminator import *
import cv2
from mtutils import min_max_normalize
import random

import random
from math import sqrt
import numpy as np

class SpaRandomization(nn.Module):
    def __init__(self, num_features, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, ):
        N, C, H, W = x.size()
        # x = self.norm(x)
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)  
            mean = 0.5 * mean + 0.5 * mean[idx_swap]  
            var = 0.5 * var + 0.5 * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x, idx_swap


class Generator_double_branchV3_noLE(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)

        self.conv2 = nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)

        self.d_conv2 = nn.ConvTranspose2d(dim2, dim1, kernel_size=1, stride=1, padding=0)
        self.d_mp1 = nn.ConvTranspose2d(dim1, dim1, kernel_size=3, stride=3, padding=0)
        self.d_conv1 = nn.ConvTranspose2d(dim1, imdim, kernel_size=5, stride=1, padding=0)

        self.outR = nn.Conv2d(imdim, imdim, kernel_size=1, stride=1, padding=0)

        # for S
        self.conv3 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp2 = nn.MaxPool2d(3)

        self.conv4 = nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)

        self.d_conv4 = nn.ConvTranspose2d(dim2, dim1, kernel_size=1, stride=1, padding=0)
        self.d_mp2 = nn.ConvTranspose2d(dim1, dim1, kernel_size=3, stride=3, padding=0)
        self.d_conv3 = nn.ConvTranspose2d(dim1, imdim, kernel_size=5, stride=1, padding=0)

        self.outS = nn.Conv2d(imdim, imdim, kernel_size=1, stride=1, padding=0)

        self.spaRandom = SpaRandomization(imdim, device=device)

    def forward(self, x):
        in_size = x.size(0)

        r = self.relu(self.bn1(self.conv1(x)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        f_r = r
        f_r_v = f_r.view(in_size, -1)

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = torch.sigmoid(self.outR(r))

        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        f_s = s
        f_s_v = f_s.view(in_size, -1)

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        x_ss, idx_swap = self.spaRandom(x_s)  

        return x_r, x_s, x_ss, x_r + x_s, x_r + x_ss, f_r_v, f_s_v






