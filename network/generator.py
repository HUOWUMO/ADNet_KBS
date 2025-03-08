import torch
import torch.nn as nn
import torch.nn.functional as F
from .discriminator import *
import cv2
from mtutils import min_max_normalize
import random
from .morph_layers2D_torch import *
from network.style_hallucination import StyleHallucination

from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
import random
from math import sqrt
import numpy as np


class MorphNet(nn.Module):
    def __init__(self, inchannel):
        super(MorphNet, self).__init__()
        num = 1
        kernel_size = 3
        self.conv1 = nn.Conv2d(inchannel, num, kernel_size=1, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.Erosion2d_1 = Erosion2d(num, num, kernel_size, soft_max=False)
        self.Dilation2d_1 = Dilation2d(num, num, kernel_size, soft_max=False)
        self.Erosion2d_2 = Erosion2d(num, num, kernel_size, soft_max=False)
        self.Dilation2d_2 = Dilation2d(num, num, kernel_size, soft_max=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        xop_2 = self.Dilation2d_1(self.Erosion2d_1(x))  # 带可学习偏置的形态学开闭运算
        xcl_2 = self.Erosion2d_2(self.Dilation2d_2(x))
        x_top = x - xop_2  # 残差连接，提前给出输入信息
        x_blk = xcl_2 - x
        x_morph = torch.cat((x_top, x_blk, xop_2, xcl_2), 1)
        # https://blog.csdn.net/qq_41731861/article/details/123919662
        return x_morph


class SpeRandomization(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, idx_swap, y=None):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()
            if y != None:
                for i in range(len(y.unique())):
                    index = y == y.unique()[i]
                    tmp, mean_tmp, var_tmp = x[index], mean[index], var[index]
                    tmp = tmp[torch.randperm(tmp.size(0))].detach()
                    tmp = tmp * (var_tmp + self.eps).sqrt() + mean_tmp
                    x[index] = tmp
            else:
                # idx_swap = torch.randperm(N)
                x = x[idx_swap].detach()

                x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)
        return x


class SpeRandomization_InternalSwap(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, y=None):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()
            if y != None:
                for i in range(len(y.unique())):
                    index = y == y.unique()[i]
                    tmp, mean_tmp, var_tmp = x[index], mean[index], var[index]
                    tmp = tmp[torch.randperm(tmp.size(0))].detach()
                    tmp = tmp * (var_tmp + self.eps).sqrt() + mean_tmp
                    x[index] = tmp
            else:
                idx_swap = torch.randperm(N)
                x = x[idx_swap].detach()

                x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)
        return x


class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        # self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * x + beta
        # return (1+gamma)*(x)+beta


class STN(nn.Module):
    def __init__(self, imdim, imsize, class_num):
        super(STN, self).__init__()
        self.zdim = class_num * imsize * imsize
        self.localization = nn.Sequential(nn.Conv2d(imdim, 8, kernel_size=5, padding=2),
                                          nn.ReLU(True),
                                          nn.Conv2d(8, class_num, kernel_size=3, padding=1),
                                          nn.ReLU(True))
        self.fc_loc = nn.Sequential(nn.Linear(self.zdim, 32),
                                    nn.ReLU(True),
                                    nn.Linear(32, 3 * 2))
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.zdim)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class Generator_SSECNet(nn.Module):
    def __init__(self, num_class, n=64, kernelsize=3, imdim=48, imsize=13, zdim=16, device=0):
        super().__init__()
        stride = (kernelsize - 1) // 2
        self.zdim = zdim
        self.imdim = imdim
        self.imsize = imsize
        self.device = device
        self.zdim = n

        # 提取空间信息
        self.conv_spa1 = nn.Conv2d(imdim, 3, 1, 1)
        self.conv_spa2 = nn.Conv2d(3, n, 1, 1)
        # 使用STN改变content
        self.stn = STN(3, imsize[0], num_class)
        # 提取光谱信息
        self.conv_spe1 = nn.Conv2d(imdim, n, imsize[0], 1)
        self.conv_spe2 = nn.ConvTranspose2d(n, n, imsize[0])
        # 使用AdaIN改变style
        self.adain = AdaIN2d(n, n)

        # 还原维度
        self.conv1_stn = nn.Conv2d(n, n, kernelsize, 1, stride)
        self.conv2_stn = nn.Conv2d(n, imdim, kernelsize, 1, stride)
        self.conv1_adain = nn.Conv2d(n, n, kernelsize, 1, stride)
        self.conv2_adain = nn.Conv2d(n, imdim, kernelsize, 1, stride)

    def forward(self, x):
        x_spa = self.conv_spa1(x)
        x_spe = self.conv_spe1(x)

        z = torch.randn(len(x), self.zdim).to(self.device)
        x_adain = self.adain(x_spe, z)
        x_adain = self.conv_spe2(x_adain)

        x_stn = self.stn(x_spa)
        x_stn = self.conv_spa2(x_stn)

        x_stn = F.relu(self.conv1_stn(x_stn))
        x_stn = torch.sigmoid(self.conv2_stn(x_stn))

        x_adain = F.relu(self.conv1_adain(x_adain))
        x_adain = torch.sigmoid(self.conv2_adain(x_adain))

        return x_adain, x_stn


class SpaRandomization_CrossInC(nn.Module):  # 在不同的特征通道间cross
    def __init__(self, num_features, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(
            device)  # requires_grad 不管是False还是Ture都不影响IDSEnet的训练,self.alpha直接改0.5也不影响

    def forward(self, x, ):
        N, C, H, W = x.size()
        # x = self.norm(x)
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(C)

            # https://blog.csdn.net/qq_45100200/article/details/123247582 Pytorch 按某个维度打乱数据方法
            mean = self.alpha * mean + (1 - self.alpha) * mean[:, idx_swap, :]  # 和固定成0.5相比，训练结果没有区别
            var = self.alpha * var + (1 - self.alpha) * var[:, idx_swap, :]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class Generator(nn.Module):
    def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[13, 13], zdim=10, device=0):
        ''' w_ln 局部噪声权重
        '''
        super().__init__()
        stride = (kernelsize - 1) // 2
        self.zdim = zdim  # AdaIN2d随机正则化的全连接层配置
        self.imdim = imdim
        self.imsize = imsize
        self.device = device
        num_morph = 4
        self.Morphology = MorphNet(imdim)
        self.adain2_morph = AdaIN2d(zdim, num_morph)

        self.conv_spa1 = nn.Conv2d(imdim, 3, 1, 1)
        self.conv_spa2 = nn.Conv2d(3, n, 1, 1)
        self.conv_spe1 = nn.Conv2d(imdim, n, imsize[0], 1)
        self.conv_spe2 = nn.ConvTranspose2d(n, n, imsize[0])
        self.conv1 = nn.Conv2d(n + n + num_morph, n, kernelsize, 1, stride)
        self.conv2 = nn.Conv2d(n, imdim, kernelsize, 1, stride)

        self.speRandom = SpeRandomization(n)
        self.spaRandom = SpaRandomization(3, device=device)

    def forward(self, x):
        x_morph = self.Morphology(x)
        z = torch.randn(len(x), self.zdim).to(self.device)
        x_morph = self.adain2_morph(x_morph, z)  # 迁移 （噪声->全连接->）风格 到x_morph

        x_spa = F.relu(self.conv_spa1(x))
        x_spe = F.relu(self.conv_spe1(x))
        x_spa, idx_swap = self.spaRandom(x_spa)
        x_spe = self.speRandom(x_spe, idx_swap)
        x_spe = self.conv_spe2(x_spe)
        x_spa = self.conv_spa2(x_spa)
        x = F.relu(self.conv1(torch.cat((x_spa, x_spe, x_morph), 1)))
        x = torch.sigmoid(self.conv2(x))

        return x


class SpaRandomization(nn.Module):
    def __init__(self, num_features, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        # InstanceNorm： https://zhuanlan.zhihu.com/p/395855181
        # 一个channel内做归一化，算H*W的均值，用在风格化迁移；因为在图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，
        # 因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。

        # SpaRandomization在InstanceNorm基础之上，在mini-batch内部将mean、var进行了打乱两两加权，打破了每个图像实例之间的独立，让不同图像风格可以相互引导融合

        # 对比BN在一个batch内归一化：https://blog.csdn.net/ECNU_LZJ/article/details/104203604
        # 把网络的每个隐含层的分布都归一化到标准正态。其实就是把越来越偏的分布强制拉回到比较标准的分布，这样使得激活函数的输入值落在该激活函数对输入比较敏感的区域，
        # 这样一来输入的微小变化就会导致损失函数较大的变化。通过这样的方式可以使梯度变大，就避免了梯度消失的问题，而且梯度变大意味着收敛速度快，能大大加快训练速度。
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(
            device)  # requires_grad 不管是False还是Ture都不影响IDSEnet的训练,self.alpha直接改0.5也不影响

    def forward(self, x, ):
        N, C, H, W = x.size()
        # x = self.norm(x)
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)  # 别注释，会改变初始化
            mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]  # 和固定成0.5相比，训练结果没有区别
            var = self.alpha * var + (1 - self.alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x, idx_swap


#
class Perturbation(nn.Module):
    def __init__(self, num_features, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(
            device)  # requires_grad 不管是False还是Ture都不影响IDSEnet的训练,self.alpha直接改0.5也不影响

    def forward(self, x, ):
        N, C, H, W = x.size()
        # x = self.norm(x)
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            std = (var + self.eps).sqrt()

            x = (x - mean) / std

            # 将高斯噪声注入到单个源域图像的特征通道统计中来合成各种域样式
            # https://blog.csdn.net/qq_52589927/article/details/142794923
            # Perturbation
            rand_mean = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.5))
            rand_var = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.5))

            mean = mean + rand_mean * mean
            std = std + rand_var * std  # 注意扰动的是标准差，如果扰动方差，容易Loss=Nan

            x = x * std + mean
            x = x.view(N, C, H, W)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8,
                 mask_radio=0.1, mask_alpha=0.5,
                 noise_mode=1,
                 low_or_high=0, uncertainty_model=0, perturb_prob=0.5,
                 uncertainty_factor=1.0,
                 noise_layer_flag=0, gauss_or_uniform=0, ):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w - (h // 2), dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

        self.mask_radio = mask_radio

        self.noise_mode = noise_mode
        self.noise_layer_flag = noise_layer_flag

        self.alpha = mask_alpha

        self.low_or_high = low_or_high

        self.eps = 1e-6
        self.factor = uncertainty_factor
        self.uncertainty_model = uncertainty_model
        self.p = perturb_prob
        self.gauss_or_uniform = gauss_or_uniform

    def _reparameterize(self, mu, std, epsilon_norm):
        # epsilon = torch.randn_like(std) * self.factor
        epsilon = epsilon_norm * self.factor
        mu_t = mu + epsilon * std
        return mu_t

    def spa_noise(self, img_fft, ratio=1.0, noise_mode=1,
                  low_or_high=0, uncertainty_model=0, gauss_or_uniform=0):
        """Input image size: ndarray of [H, W, C]"""
        """noise_mode: 1 amplitude; 2: phase 3:both"""
        """uncertainty_model: 1 batch-wise modeling 2: channel-wise modeling 3:token-wise modeling"""
        if random.random() > self.p:
            return img_fft
        batch_size, h, w, c = img_fft.shape

        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)

        if low_or_high == 0:
            img_abs = torch.fft.fftshift(img_abs, dim=(1, 2))

        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = w - w_crop

        img_abs_ = img_abs.clone()
        if noise_mode != 0:
            if uncertainty_model != 0:
                if uncertainty_model == 1:
                    # batch level modeling
                    miu = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:, :], dim=(1, 2), keepdim=True)
                    var = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:, :], dim=(1, 2), keepdim=True)
                    sig = (var + self.eps).sqrt()  # Bx1x1xC

                    var_of_miu = torch.var(miu, dim=0, keepdim=True)
                    var_of_sig = torch.var(sig, dim=0, keepdim=True)
                    sig_of_miu = (var_of_miu + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)
                    sig_of_sig = (var_of_sig + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)  # Bx1x1xC

                    if gauss_or_uniform == 0:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)

                        miu_mean = miu
                        sig_mean = sig

                        beta = self._reparameterize(mu=miu_mean, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig_mean, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    elif gauss_or_uniform == 1:
                        epsilon_norm_miu = torch.rand_like(sig_of_miu) * 2 - 1.  # U(-1,1)
                        epsilon_norm_sig = torch.rand_like(sig_of_sig) * 2 - 1.
                        beta = self._reparameterize(mu=miu, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    else:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)
                        beta = self._reparameterize(mu=miu, std=1., epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=1., epsilon_norm=epsilon_norm_sig)

                    # adjust statistics for each sample
                    img_abs[:, h_start:h_start + h_crop, w_start:, :] = gamma * (
                            img_abs[:, h_start:h_start + h_crop, w_start:, :] - miu) / sig + beta

                elif uncertainty_model == 2:
                    # element level modeling
                    miu_of_elem = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:, :], dim=0, keepdim=True)
                    var_of_elem = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:, :], dim=0, keepdim=True)
                    sig_of_elem = (var_of_elem + self.eps).sqrt()  # 1xHxWxC

                    if gauss_or_uniform == 0:
                        epsilon_sig = torch.randn_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:, :])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    elif gauss_or_uniform == 1:
                        epsilon_sig = torch.rand_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:, :]) * 2 - 1.  # U(-1,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    else:
                        epsilon_sig = torch.randn_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:, :])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * self.factor

                    img_abs[:, h_start:h_start + h_crop, w_start:, :] = img_abs[:, h_start:h_start + h_crop, w_start:,
                                                                        :] + gamma
        else:
            pass
        if low_or_high == 0:
            img_abs = torch.fft.ifftshift(img_abs, dim=(1, 2))  # recover

        img_mix = img_abs * (np.e ** (1j * img_pha))
        return img_mix

    def forward(self, x, layer_index=0, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.training:
            if self.noise_mode != 0 and self.noise_layer_flag == 1:
                x = self.spa_noise(x, ratio=self.mask_radio, noise_mode=self.noise_mode,
                                   uncertainty_model=self.uncertainty_model,
                                   gauss_or_uniform=self.gauss_or_uniform)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        return x


class BlockLayerScale(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5,
                 mask_radio=0.1, mask_alpha=0.5, noise_mode=1, low_or_high=0,
                 uncertainty_model=0, perturb_prob=0.5, uncertainty_factor=1.0,
                 layer_index=0, noise_layers=[0, 1, 2, 3], gauss_or_uniform=0, ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if layer_index in noise_layers:
            noise_layer_flag = 1
        else:
            noise_layer_flag = 0
        self.filter = GlobalFilter(dim, h=h, w=w,
                                   mask_radio=mask_radio,
                                   mask_alpha=mask_alpha,
                                   noise_mode=noise_mode,
                                   low_or_high=low_or_high, uncertainty_model=uncertainty_model,
                                   perturb_prob=perturb_prob,
                                   uncertainty_factor=uncertainty_factor,
                                   noise_layer_flag=noise_layer_flag, gauss_or_uniform=gauss_or_uniform, )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        self.layer_index = layer_index  # where is the block in

    def forward(self, input):
        x = input

        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x), self.layer_index))))
        return x


# 3D-空谱特征随机化
class Spa_Spe_Randomization(nn.Module):
    def __init__(self, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)  # 定义一个可学习的参数，并初始化

    def forward(self, x, ):
        N, C, L, H, W = x.size()
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)  # 随机数是怎么改变的？每一次前向传播都会重新洗牌
            mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]  # 从batch中选择随机化均值和方差
            var = self.alpha * var + (1 - self.alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, L, H, W)

        return x, idx_swap


class Generator_3DCNN_SupCompress_pca(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], device=0, dim1=128, dim2=8):
        super().__init__()

        self.patch_size = imsize[0]

        self.n_channel = dim2
        self.n_pca = dim1

        # 2D_CONV
        self.conv_pca = nn.Conv2d(imdim, self.n_pca, 1, 1)  # 本质上是对光谱维度的下采样线性插值，还原出原始光谱
        self.inchannel = self.n_pca
        # self.inchannel = imdim  # 用于no linear实验

        # 3D_CONV
        self.conv1 = nn.Conv3d(in_channels=1,
                               out_channels=self.n_channel,
                               kernel_size=(3, 3, 3))
        # 3D空谱随机化
        self.Spa_Spe_Random = Spa_Spe_Randomization(device=device)

        # 2D空间随机化
        # self.spaRandom = SpaRandomization(self.n_pca, device=device)  # 用于no Spa-Spe Joint实验,不用的时候注释掉

        # 3D_CONV-在kernel_size不变下,还原和3D_CONV成一个逆过程(期望结构上的逆过程能够约束数据的采样还原)
        self.conv6 = nn.ConvTranspose3d(in_channels=self.n_channel, out_channels=1, kernel_size=(3, 3, 3))

        # 2D_CONV
        self.conv_inverse_pca = nn.Conv2d(self.n_pca, imdim, 1, 1)

    def forward(self, x):
        x = self.conv_pca(x)

        x = x.reshape(-1, self.patch_size, self.patch_size, self.inchannel, 1)  # (256,48,13,13,1)转换输入size,适配Conv3d输入
        x = x.permute(0, 4, 3, 1, 2)  # (256,1,48,13,13)
        x = F.relu(self.conv1(x))

        x, idx_swap = self.Spa_Spe_Random(x)
        # x, idx_swap = self.spaRandom(x)  # 用于no Spa-Spe Joint实验

        x = torch.sigmoid(self.conv6(x))
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(-1, self.inchannel, self.patch_size, self.patch_size)

        x = self.conv_inverse_pca(x)
        return x


# 如果该网络真要做本征分解任务，表达能力可能不足
class Generator_double_branch(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=64, dim2=32, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(dim1, dim2, kernel_size=3, stride=1, padding=0)

        self.conv3 = nn.ConvTranspose2d(dim2, dim1, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.ConvTranspose2d(dim1, imdim, kernel_size=5, stride=1, padding=0)

        # for S
        self.conv5 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.conv6 = nn.Conv2d(dim1, dim2, kernel_size=3, stride=1, padding=0)

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossChannel
        self.spaRandom_CrossInC = SpaRandomization_CrossInC(dim2, device=device)

        self.perturbation = Perturbation(dim2, device=device)

        # CrossChannel
        # self.spaRandom = SpaRandomization_CrossInC(dim2, device=device)

        # StyleHallucination
        self.SHM = StyleHallucination(concentration_coeff=0.0156, base_style_num=dim2, style_dim=dim2)

        self.conv7 = nn.ConvTranspose2d(dim2, dim1, kernel_size=3, stride=1, padding=0)
        self.conv8 = nn.ConvTranspose2d(dim1, imdim, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)  # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.relu(self.conv2(out1))

        # out2, idx_swap = self.spaRandom(out2) # 实例洗牌 （会导致OA下降）

        # out2 = self.spaRandom_CrossInC(out2) # 通道洗牌

        # out2 = self.perturbation(out2)  # Perturbation

        out3 = self.relu(self.conv3(out2))
        x_r = self.conv4(out3)
        # --------------------------------------------------------------------------------
        out4 = self.conv5(x)
        out5 = self.relu(self.conv6(out4))

        out5, idx_swap = self.spaRandom(out5)  # IDSEnet原版

        # out5 = self.spaRandom(out5)  # For CrossChannel

        # 风格幻觉 [作为图像/特征增强结构，适用于训练后增强非训练中增强]
        # out5, out5_style = self.SHM(out5)
        # out5 = torch.cat((out5, out5_style), dim=0)

        out6 = self.relu(self.conv7(out5))
        x_s = torch.sigmoid(self.conv8(out6))

        return x_r, x_s, x_r * x_s


class Generator_double_branchRes(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=64, dim2=32, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, imdim, kernel_size=5, stride=1, padding=2)  # ResNet注意kernel_size对应padding
        self.conv1_1 = nn.Conv2d(imdim, imdim, kernel_size=5, stride=1, padding=2)

        self.conv2 = nn.Conv2d(imdim, imdim, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(imdim, imdim, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.ConvTranspose2d(imdim, imdim, kernel_size=3, stride=1, padding=1)  # 非残差解码
        self.conv4 = nn.ConvTranspose2d(imdim, imdim, kernel_size=5, stride=1, padding=2)

        # for S
        self.conv5 = nn.Conv2d(imdim, imdim, kernel_size=5, stride=1, padding=2)
        self.conv5_1 = nn.Conv2d(imdim, imdim, kernel_size=5, stride=1, padding=2)

        self.conv6 = nn.Conv2d(imdim, imdim, kernel_size=3, stride=1, padding=1)
        self.conv6_1 = nn.Conv2d(imdim, imdim, kernel_size=3, stride=1, padding=1)

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(imdim, device=device)

        self.conv7 = nn.ConvTranspose2d(imdim, imdim, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.ConvTranspose2d(imdim, imdim, kernel_size=5, stride=1, padding=2)

        self.relu = nn.ReLU(inplace=True)  # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存

    def forward(self, x):
        out1 = self.conv1_1(self.relu(self.conv1(x)))
        out1 = x + out1
        out2 = self.conv2_1(self.relu(self.conv2(out1)))
        out2 = out2 + out1

        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))

        x_r = out4
        # --------------------------------------------------------------------------------
        out5 = self.conv5_1(self.relu(self.conv5(x)))
        out5 = x + out5
        out6 = self.conv6_1(self.relu(self.conv6(out5)))
        out6 = out6 + out5

        out6, idx_swap = self.spaRandom(out6)  # IDSEnet原版

        out7 = self.relu(self.conv7(out6))
        out8 = self.relu(self.conv8(out7))

        x_s = torch.sigmoid(out8)

        return x_r, x_s, x_r * x_s


class Generator_double_branch_CBM2D(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

        # self.SpaRandomization_CrossInC = SpaRandomization_CrossInC(imdim, device=device)  # ChannelShuffle
        # self.perturbation = Perturbation(imdim, device=device)
        # self.speRandom = SpeRandomization_InternalSwap(imdim)

    def forward(self, x):
        in_size = x.size(0)

        r = self.relu(self.bn1(self.conv1(x)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = self.outR(r)

        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        # s = self.SpaRandomization_CrossInC(s)  # Channel Shuffle
        # s = self.perturbation(s)  # Channel Perturbation
        # s = self.speRandom(s)  # Spectral Random

        s, idx_swap = self.spaRandom(s)  # 丰富S

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        return x_r, x_s, x_r * x_s


class Generator_double_branch_CBM2D_FreAug(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

        # FreAug
        self.low_freq_spa = BlockLayerScale(dim=imdim, h=imsize[0], w=imsize[1], uncertainty_model=2)

        # fusion
        self.convFusion = nn.Conv2d(2 * imdim, imdim, 3, 1, 1)

    def forward(self, x):
        in_size = x.size(0)

        r = self.relu(self.bn1(self.conv1(x)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = self.outR(r)

        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        s, idx_swap = self.spaRandom(s)  # 丰富S

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        # ------------------------------------------------------------------------------
        B, C, H, W = x.shape
        x1 = x.view(B, C, -1)
        x1 = x1.transpose(1, 2)
        x_Fre = self.low_freq_spa(x1)
        x_Fre = x_Fre.transpose(1, 2)
        x_Fre = x_Fre.view(B, C, H, W)

        # -----------------------------------------------------------------------------
        x_Fre_r = torch.cat((x_Fre, x_r), dim=1)
        x_fusion = self.convFusion(x_Fre_r)

        return x_r, x_s, x_Fre, x_fusion, x_fusion * x_s


class Generator_double_branch_CBM2D_FreSer(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

        # FreAug
        self.low_freq_spa = BlockLayerScale(dim=imdim, h=imsize[0], w=imsize[1], uncertainty_model=2)

        # fusion
        self.convFusion = nn.Conv2d(2 * imdim, imdim, 3, 1, 1)

    def forward(self, x):
        in_size = x.size(0)

        # ------------------------------------------------------------------------------
        B, C, H, W = x.shape
        x1 = x.view(B, C, -1)
        x1 = x1.transpose(1, 2)
        x_Fre = self.low_freq_spa(x1)
        x_Fre = x_Fre.transpose(1, 2)
        x_Fre = x_Fre.view(B, C, H, W)

        # ------------------------------------------------------------------------
        r = self.relu(self.bn1(self.conv1(x_Fre)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = self.outR(r)
        # --------------------------------------------------------------------------------
        s = self.relu(self.bn3(self.conv3(x_Fre)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        s, idx_swap = self.spaRandom(s)  # 丰富S

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        return x_Fre, x_r, x_s, x_r * x_s


class Generator_double_branch_CBM2D_FreSerRes(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

        # FreAug
        self.low_freq_spa = BlockLayerScale(dim=imdim, h=imsize[0], w=imsize[1], uncertainty_model=2)

    def forward(self, x):
        in_size = x.size(0)

        # ------------------------------------------------------------------------------
        B, C, H, W = x.shape
        x1 = x.view(B, C, -1)
        x1 = x1.transpose(1, 2)
        x_Fre = self.low_freq_spa(x1)
        x_Fre = x_Fre.transpose(1, 2)
        x_Fre = x_Fre.view(B, C, H, W)

        # ------------------------------------------------------------------------
        r = self.relu(self.bn1(self.conv1(x_Fre + x)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = self.outR(r)
        # --------------------------------------------------------------------------------
        s = self.relu(self.bn3(self.conv3(x_Fre + x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        s, idx_swap = self.spaRandom(s)  # 丰富S

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        return x_r, x_s, x_r * x_s


class Generator_double_branch_CBM2D_FreSerDouble(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

        # FreAug
        self.low_freq_spa = BlockLayerScale(dim=imdim, h=imsize[0], w=imsize[1], uncertainty_model=2)

        # FreAug2
        self.low_freq_spa2 = BlockLayerScale(dim=imdim, h=imsize[0], w=imsize[1], uncertainty_model=2)

        # fusion
        self.convFusion = nn.Conv2d(2 * imdim, imdim, 3, 1, 1)

    def forward(self, x):
        in_size = x.size(0)

        # ------------------------------------------------------------------------------
        B, C, H, W = x.shape
        x1 = x.view(B, C, -1)
        x1 = x1.transpose(1, 2)

        x_Fre = self.low_freq_spa(x1)
        x_Fre = x_Fre.transpose(1, 2)
        x_Fre = x_Fre.view(B, C, H, W)

        x_Fre2 = self.low_freq_spa2(x1)
        x_Fre2 = x_Fre2.transpose(1, 2)
        x_Fre2 = x_Fre2.view(B, C, H, W)

        # ------------------------------------------------------------------------
        r = self.relu(self.bn1(self.conv1(x_Fre)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = self.outR(r)

        # --------------------------------------------------------------------------------
        s = self.relu(self.bn3(self.conv3(x_Fre2)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        s, idx_swap = self.spaRandom(s)  # 丰富S

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        return x_r, x_s, x_r * x_s


class Generator_double_branch_CBM2D_FreSerConv(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

        # FreAug
        self.low_freq_spa = BlockLayerScale(dim=imdim, h=imsize[0], w=imsize[1], uncertainty_model=2)

        # 模仿FDGNET前后线性层
        self.conv_in = nn.Conv2d(imdim, imdim, 3, 1, 1)
        self.conv_out = nn.Conv2d(imdim, imdim, 3, 1, 1)

    def forward(self, x):
        in_size = x.size(0)

        # ------------------------------------------------------------------------------
        x = self.conv_in(x)

        B, C, H, W = x.shape
        x1 = x.view(B, C, -1)
        x1 = x1.transpose(1, 2)
        x_Fre = self.low_freq_spa(x1)
        x_Fre = x_Fre.transpose(1, 2)
        x_Fre = x_Fre.view(B, C, H, W)

        x_Fre = self.conv_out(x_Fre)

        # ------------------------------------------------------------------------
        r = self.relu(self.bn1(self.conv1(x_Fre)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = self.outR(r)

        # --------------------------------------------------------------------------------
        s = self.relu(self.bn3(self.conv3(x_Fre)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        s, idx_swap = self.spaRandom(s)  # 丰富S

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        return x_r, x_s, x_r * x_s


class Generator_double_branch_CBM2D_FreAlign(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

        # FreAug
        self.low_freq_spa = BlockLayerScale(dim=imdim, h=imsize[0], w=imsize[1], uncertainty_model=2)

    def forward(self, x):
        in_size = x.size(0)

        r = self.relu(self.bn1(self.conv1(x)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = self.outR(r)

        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        s, idx_swap = self.spaRandom(s)  # 丰富S

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        # ------------------------------------------------------------------------------
        B, C, H, W = x.shape
        x1 = x.view(B, C, -1)
        x1 = x1.transpose(1, 2)
        x_Fre = self.low_freq_spa(x1)
        x_Fre = x_Fre.transpose(1, 2)
        x_Fre = x_Fre.view(B, C, H, W)

        return x_r, x_s, x_Fre, x_r * x_s


class Generator_double_branch_CBM2D_FreAugAdd(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

        # FreAug
        self.low_freq_spa = BlockLayerScale(dim=imdim, h=imsize[0], w=imsize[1], uncertainty_model=2)

        # fusion
        self.convFusion = nn.Conv2d(2 * imdim, imdim, 3, 1, 1)

    def forward(self, x):
        in_size = x.size(0)

        r = self.relu(self.bn1(self.conv1(x)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = self.outR(r)

        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        s, idx_swap = self.spaRandom(s)  # 丰富S

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        # ------------------------------------------------------------------------------
        B, C, H, W = x.shape
        x1 = x.view(B, C, -1)
        x1 = x1.transpose(1, 2)
        x_Fre = self.low_freq_spa(x1)
        x_Fre = x_Fre.transpose(1, 2)
        x_Fre = x_Fre.view(B, C, H, W)

        # -----------------------------------------------------------------------------
        x_fusion = x_Fre + x_r

        return x_r, x_s, x_Fre, x_fusion, x_fusion * x_s


class Generator_double_branch_CBM2D_FreIntr(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

        # FreAug
        self.low_freq_spa = BlockLayerScale(dim=imdim, h=imsize[0], w=imsize[1], uncertainty_model=2)

    def forward(self, x):
        s = self.relu(self.bn3(self.conv3(x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        s, idx_swap = self.spaRandom(s)  # 丰富S

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        # ------------------------------------------------------------------------------
        B, C, H, W = x.shape
        x1 = x.view(B, C, -1)
        x1 = x1.transpose(1, 2)
        x_Fre = self.low_freq_spa(x1)
        x_Fre = x_Fre.transpose(1, 2)
        x_Fre = x_Fre.view(B, C, H, W)

        return x_Fre, x_s, x_Fre * x_s


class Generator_double_branch_CBM2D_Fre(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

        # FreAug
        self.low_freq_spa = BlockLayerScale(dim=imdim, h=imsize[0], w=imsize[1], uncertainty_model=2)

        # fusion
        self.convFusion = nn.Conv2d(2 * imdim, imdim, 3, 1, 1)

    def forward(self, x):
        in_size = x.size(0)

        r = self.relu(self.bn1(self.conv1(x)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = self.outR(r)

        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        s, idx_swap = self.spaRandom(s)  # 丰富S

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        # ------------------------------------------------------------------------------
        B, C, H, W = x.shape
        x1 = x.view(B, C, -1)
        x1 = x1.transpose(1, 2)
        x_Fre = self.low_freq_spa(x1)
        x_Fre = x_Fre.transpose(1, 2)
        x_Fre = x_Fre.view(B, C, H, W)

        # -----------------------------------------------------------------------------
        x_Fre_r = torch.cat((x_Fre, x_r), dim=1)
        x_fusion = self.convFusion(x_Fre_r)

        return x_r, x_s, x_Fre, x_fusion


class Generator_double_branch_CBM2D_Whiten(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

    def forward(self, x):
        in_size = x.size(0)

        r = self.relu(self.bn1(self.conv1(x)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = self.outR(r)

        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))
        s_in = s
        s, idx_swap = self.spaRandom(s)  # 丰富S
        s_Wout = s
        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        return x_r, x_s, x_r * x_s, s_in, s_Wout


class Generator_double_branch_ATTEN(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(dim2, device=device)

        # 潜在特征提取
        self.LE = nn.Conv2d(dim2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        in_size = x.size(0)

        r = self.relu(self.bn1(self.conv1(x)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        f_r = self.LE(r)
        f_r_v = f_r.view(in_size, -1)

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = self.outR(r)

        # x_rr, idx_swap = self.spaRandom(x_r)  # 丰富R
        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        f_s = self.LE(s)
        f_s_v = f_s.view(in_size, -1)

        s, idx_swap = self.spaRandom(s)

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        # 这里采用图像分解约束，非本征分解约束
        return x_r, x_s, x_s, x_r * x_s, f_r_v, f_s_v


class Generator_double_branchV2(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

        self.conv2 = nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)

        self.d_conv2 = nn.ConvTranspose2d(dim2, dim1, kernel_size=1, stride=1, padding=0)
        self.d_mp = nn.ConvTranspose2d(dim1, dim1, kernel_size=3, stride=3, padding=0)
        self.d_conv1 = nn.ConvTranspose2d(dim1, imdim, kernel_size=5, stride=1, padding=0)

        self.outR = nn.Conv2d(imdim, imdim, kernel_size=1, stride=1, padding=0)

        # for S
        self.conv3 = nn.Conv2d(imdim, dim1, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)

        self.d_conv4 = nn.ConvTranspose2d(dim2, dim1, kernel_size=1, stride=1, padding=0)
        self.d_conv3 = nn.ConvTranspose2d(dim1, imdim, kernel_size=3, stride=1, padding=0)

        self.outS = nn.Conv2d(imdim, imdim, kernel_size=1, stride=1, padding=0)

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(imdim, device=device)

    def forward(self, x):
        r = self.relu(self.bn1(self.conv1(x)))
        r = self.mp(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp(self.d_conv2(r)))
        x_r = torch.sigmoid(self.outR(r))

        # x_rr, idx_swap = self.spaRandom(x_r)  # 丰富R
        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.relu(self.bn4(self.conv4(s)))

        s = self.d_conv3(self.d_conv4(s))
        x_s = self.outS(s)

        x_ss, idx_swap = self.spaRandom(x_s)  # 丰富S
        # x_ss = x_s  # no shuffle实验
        return x_r, x_s, x_ss, x_r * x_s, x_r * x_ss


class Generator_VAE(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, latent_dims=128, device=0):
        super().__init__()

        self.patch_size = imsize[0]
        self.dim2 = dim2
        self.imdim = imdim
        self.device = device

        # ---避免_get_size()报错
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)
        self.mp = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(dim2)

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim2),
            nn.ReLU(inplace=True)
        )

        self.c, self.w, self.h = self._get_size()
        self.dim_expand = self.dim2 * self.w * self.h

        self.mu = nn.Linear(self.dim_expand, latent_dims)
        self.logvar = nn.Linear(self.dim_expand, latent_dims)  # 由于方差是非负的，因此预测方差对数

        # decoder
        self.recover = nn.Linear(latent_dims, self.dim_expand)

        self.d_conv2 = nn.ConvTranspose2d(dim2, dim1, kernel_size=1, stride=1, padding=0)
        self.d_mp = nn.ConvTranspose2d(dim1, dim1, kernel_size=3, stride=3, padding=0)
        self.d_conv1 = nn.ConvTranspose2d(dim1, imdim, kernel_size=5, stride=1, padding=0)

        self.out = nn.Conv2d(imdim, imdim, kernel_size=1, stride=1, padding=0)

    # 重参数，为了可以反向传播
    def reparametrization(self, mu, logvar):
        # sigma = exp(0.5 * log(sigma^2))= exp(0.5 * log(var))
        std = torch.exp(0.5 * logvar)
        # N(mu, std^2) = N(0, 1) * std + mu
        z = torch.randn(std.size(), device=mu.device) * std + mu
        return z

    def _get_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.imdim, self.patch_size, self.patch_size))
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.mp(x)
            x = self.relu(self.bn2(self.conv2(x)))
            b, c, w, h = x.size()
        return c, w, h

    def forward(self, x):
        in_size = x.size(0)

        en = self.encoder(x)
        mu = self.mu(en.view(in_size, -1))
        logvar = self.logvar(en.view(in_size, -1))
        z = self.reparametrization(mu, logvar)

        # decoder
        out = self.recover(z)
        out = out.view(in_size, self.dim2, self.w, self.h)
        out = self.d_conv1(self.d_mp(self.d_conv2(out)))
        out = torch.sigmoid(self.out(out))

        return out, mu, logvar


class Generator_double_branchV3(nn.Module):  # 在V2的基础上能够增加正交约束，这里不再有R/S分量，双分支结构对称
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(imdim, device=device)  # ADNet InstanceShuffle

        # self.SpaRandomization_CrossInC = SpaRandomization_CrossInC(imdim, device=device)  # ChannelShuffle
        # self.perturbation = Perturbation(imdim, device=device)
        # self.speRandom = SpeRandomization_InternalSwap(imdim)

        # 潜在特征提取
        self.LE = nn.Conv2d(dim2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        in_size = x.size(0)

        r = self.relu(self.bn1(self.conv1(x)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        f_r = self.LE(r)
        f_r_v = f_r.view(in_size, -1)

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = torch.sigmoid(self.outR(r))

        # x_rr, idx_swap = self.spaRandom(x_r)  # 丰富R
        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        f_s = self.LE(s)
        f_s_v = f_s.view(in_size, -1)

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        x_ss, idx_swap = self.spaRandom(x_s)  # 丰富S
        # x_ss = self.SpaRandomization_CrossInC(x_s)  # 丰富S
        # x_ss = self.perturbation(x_s)  # 丰富S
        # x_ss = self.speRandom(x_s)  # 丰富S

        # x_ss = x_s  # no shuffle实验

        # 这里采用图像分解约束，非本征分解约束
        return x_r, x_s, x_ss, x_r + x_s, x_r + x_ss, f_r_v, f_s_v

        # 单分支实验
        # return x_r, x_r, x_r, x_r, x_r, f_r_v, f_s_v

        # 双分支重建，单分支扩展
        # return x_r, x_s, x_ss, x_r + x_s, x_r, f_r_v, f_s_v


class Generator_SingleR(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

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

    def forward(self, x):
        r = self.relu(self.bn1(self.conv1(x)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = torch.sigmoid(self.outR(r))

        return x_r


class Generator_double_branchV3_noLE(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

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

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
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

        # x_rr, idx_swap = self.spaRandom(x_r)  # 丰富R
        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        f_s = s
        f_s_v = f_s.view(in_size, -1)

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        x_ss, idx_swap = self.spaRandom(x_s)  # 丰富S
        # x_ss = x_s  # no shuffle实验

        # 这里采用图像分解约束，非本征分解约束
        return x_r, x_s, x_ss, x_r + x_s, x_r + x_ss, f_r_v, f_s_v

        # 单分支实验
        # return x_r, x_r, x_r, x_r, x_r, f_r_v, f_s_v

        # 双分支重建，单分支扩展
        # return x_r, x_s, x_ss, x_r + x_s, x_r, f_r_v, f_s_v


class Generator_double_branchV3_DeCorr(nn.Module):  # 为了小patch_sizw加速SVD，删除MP
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]
        self.imdim = imdim

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)

        self.d_conv2 = nn.ConvTranspose2d(dim2, dim1, kernel_size=1, stride=1, padding=0)
        self.d_conv1 = nn.ConvTranspose2d(dim1, imdim, kernel_size=5, stride=1, padding=0)

        self.outR = nn.Conv2d(imdim, imdim, kernel_size=1, stride=1, padding=0)

        # for S
        self.conv3 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)

        self.d_conv4 = nn.ConvTranspose2d(dim2, dim1, kernel_size=1, stride=1, padding=0)
        self.d_conv3 = nn.ConvTranspose2d(dim1, imdim, kernel_size=5, stride=1, padding=0)

        self.outS = nn.Conv2d(imdim, imdim, kernel_size=1, stride=1, padding=0)

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(imdim, device=device)

        # 潜在特征提取
        self.LE = nn.Conv2d(dim2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        in_size = x.size(0)

        r = self.relu(self.bn1(self.conv1(x)))
        r = self.relu(self.bn2(self.conv2(r)))

        f_r = self.LE(r)
        f_r_v = f_r.view(in_size, -1)

        r = self.d_conv1(self.d_conv2(r))
        x_r = torch.sigmoid(self.outR(r))

        # x_rr, idx_swap = self.spaRandom(x_r)  # 丰富R
        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.relu(self.bn4(self.conv4(s)))

        f_s = self.LE(s)
        f_s_v = f_s.view(in_size, -1)

        s = self.d_conv3(self.d_conv4(s))
        x_s = torch.sigmoid(self.outS(s))

        # ------------------------------------decorrlated-------------------------------------------------
        # 实验发现，通过decorrlated，训练到后期x_s均是NaN,导致SVD无法收敛
        # 存在NaN,影响
        nan_mask = torch.isnan(x_s)
        print(nan_mask)
        # fixed_value = 0.0  # 设置我们要替换 NaN 的固定值
        # x_s = torch.where(nan_mask, torch.tensor(fixed_value), x_s)  # 将 NaN 值替换为固定值

        x_s_expand = x_s.view(in_size, -1)
        x_s_expand_cen = x_s_expand - torch.mean(x_s_expand, dim=0)
        Sigma = torch.mm(x_s_expand_cen.T, x_s_expand_cen) / in_size
        # for a symmetric and positive-definite (SPD) matrix, the eigen decompositon and SCD will concide.
        U, S, VT = torch.linalg.svd(Sigma)  # we will test that U.T is euqual to VT
        # The decorrlated data
        X_decorr = torch.mm(x_s_expand_cen, U)
        x_ss = X_decorr.view(in_size, self.imdim, self.patch_size, self.patch_size)
        # -------------------------------------------------------------------------------------

        # x_ss, idx_swap = self.spaRandom(x_s)  # 丰富S
        # x_ss = x_s  # no shuffle实验

        # 这里采用图像分解约束，非本征分解约束
        return x_r, x_s, x_ss, x_r + x_s, x_r + x_ss, f_r_v, f_s_v

        # 单分支实验
        # return x_r, x_r, x_r, x_r, x_r, f_r_v, f_s_v

        # 双分支重建，单分支扩展
        # return x_r, x_s, x_ss, x_r + x_s, x_r, f_r_v, f_s_v


class Generator_double_branchV3_DeCorr_NoLE(nn.Module):  # 为了小patch_sizw加速SVD，删除MP
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]
        self.imdim = imdim

        # for R
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)

        self.d_conv2 = nn.ConvTranspose2d(dim2, dim1, kernel_size=1, stride=1, padding=0)
        self.d_conv1 = nn.ConvTranspose2d(dim1, imdim, kernel_size=5, stride=1, padding=0)

        self.outR = nn.Conv2d(imdim, imdim, kernel_size=1, stride=1, padding=0)

        # for S
        self.conv3 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)

        self.d_conv4 = nn.ConvTranspose2d(dim2, dim1, kernel_size=1, stride=1, padding=0)
        self.d_conv3 = nn.ConvTranspose2d(dim1, imdim, kernel_size=5, stride=1, padding=0)

        self.outS = nn.Conv2d(imdim, imdim, kernel_size=1, stride=1, padding=0)

        # 2D空间随机化 消除卷积的局部归纳偏执 CrossBatch
        self.spaRandom = SpaRandomization(imdim, device=device)

        # 潜在特征提取
        self.LE = nn.Conv2d(dim2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        in_size = x.size(0)

        r = self.relu(self.bn1(self.conv1(x)))
        r = self.relu(self.bn2(self.conv2(r)))

        f_r_v = r.view(in_size, -1)

        r = self.d_conv1(self.d_conv2(r))
        x_r = torch.sigmoid(self.outR(r))

        # x_rr, idx_swap = self.spaRandom(x_r)  # 丰富R
        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.relu(self.bn4(self.conv4(s)))

        f_s_v = s.view(in_size, -1)

        s = self.d_conv3(self.d_conv4(s))
        x_s = torch.sigmoid(self.outS(s))

        # ------------------------------------decorrlated-------------------------------------------------
        # 存在NaN,影响
        # nan_mask = torch.isnan(x_s)
        # fixed_value = 0.0  # 设置我们要替换 NaN 的固定值
        # x_s = torch.where(nan_mask, torch.tensor(fixed_value), x_s)  # 将 NaN 值替换为固定值

        x_s_expand = x_s.view(in_size, -1)
        x_s_expand_cen = x_s_expand - torch.mean(x_s_expand, dim=0)
        Sigma = torch.mm(x_s_expand_cen.T, x_s_expand_cen) / in_size
        # for a symmetric and positive-definite (SPD) matrix, the eigen decompositon and SCD will concide.
        U, S, VT = torch.linalg.svd(Sigma)  # we will test that U.T is euqual to VT
        # The decorrlated data
        X_decorr = torch.mm(x_s_expand_cen, U)
        x_ss = X_decorr.view(in_size, self.imdim, self.patch_size, self.patch_size)
        # -------------------------------------------------------------------------------------

        # x_ss, idx_swap = self.spaRandom(x_s)  # 丰富S
        # x_ss = x_s  # no shuffle实验

        # 这里采用图像分解约束，非本征分解约束
        return x_r, x_s, x_ss, x_r + x_s, x_r + x_ss, f_r_v, f_s_v

        # 单分支实验
        # return x_r, x_r, x_r, x_r, x_r, f_r_v, f_s_v

        # 双分支重建，单分支扩展
        # return x_r, x_s, x_ss, x_r + x_s, x_r, f_r_v, f_s_v


# 输入通过高斯核权重预处理，强化了中心像元的特征地位
class Generator_Gaussian(nn.Module):

    def __init__(self, inchannel, patch_size, std=0.2, re_grad_weight=True, init_alpha=1, re_grad_alpha=True):
        super(Generator_Gaussian, self).__init__()
        self.inchannel = inchannel
        self.patch_size = patch_size
        # 设置2D高斯权重
        gaussian_kernel_1D = cv2.getGaussianKernel(patch_size, std, cv2.CV_32F)  # 构建一维高斯核,方差为std
        gaussian_kernel_2D = gaussian_kernel_1D * gaussian_kernel_1D.T  # 由一维高斯核构建二维高斯核
        gaussian_kernel_2D = min_max_normalize(gaussian_kernel_2D)  # 归一化,整体趋势不变，数值有效增加

        kernel = torch.FloatTensor(gaussian_kernel_2D).expand(inchannel, patch_size, patch_size)  # 复制扩展高斯核
        self.weight = nn.Parameter(data=kernel, requires_grad=re_grad_weight)  #

        # 设置比例权重
        self.alpha = nn.Parameter(torch.tensor(init_alpha), requires_grad=re_grad_alpha)  # [0.0,False]等价于原始鉴别器

    def forward(self, x):  # 前向传播默认test模式

        in_size = x.size(0)  # 注意epoch末尾的batch_size<=设置的固定值，因此要重新计算

        gaussian_weight = self.weight
        gaussian_weight_expand = gaussian_weight.expand(in_size, self.inchannel, self.patch_size,
                                                        self.patch_size)  # 复制扩展高斯核
        x_gaussian = x * gaussian_weight_expand  # 点乘高斯核权重，相当于预处理

        # ----------------------用超参数混合--------------------------
        x = (1 - self.alpha) * x + self.alpha * x_gaussian

        return x


class Smooth(nn.Module):
    def __init__(self, N, ni, beta, num_iteration):
        super().__init__()

        self.N = N
        self.ni = ni
        self.beta = beta
        self.num_iteration = num_iteration

    def forward(self, y):

        # 第一轮迭代初始
        # 注意，如果不用.copy()，会直接把变量地址同化，和C++一样
        y_buff1 = y.copy()
        y_buff2 = y.copy()  # 后三个不参与迭代，直接赋予y的值

        y_buff2[0] = y_buff1[0]

        # 先移动特别的y[1]

        y_buff2[1] = y_buff1[1] - self.ni * (
                (1 - self.beta) * (-2 * y_buff1[0] + 5 * y_buff1[1] - 4 * y_buff1[2] + y_buff1[3]) + self.beta * (
                y_buff1[1] - y[1]))

        # 后面一视同仁
        for n in range(2, self.N - 2):
            y_buff2[n] = y_buff1[n] - self.ni * ((1 - self.beta) * (
                    1 * y_buff1[n - 2] - 4 * y_buff1[n - 1] + 6 * y_buff1[n] - 4 * y_buff1[n + 1] + 1 * y_buff1[
                n + 2]) + self.beta * (y_buff1[n] - y[n]))

        # 多轮迭代
        for k in range(0, self.num_iteration):

            y_buff1 = y_buff2.copy()
            y_buff2 = y.copy()

            y_buff2[0] = y_buff1[0]

            # 移动特别的y[1]
            y_buff2[1] = y_buff1[1] - self.ni * ((1 - self.beta) * (
                    -2 * y_buff1[0] + 5 * y_buff1[1] - 4 * y_buff1[2] + y_buff1[3]) + self.beta * (
                                                         y_buff1[1] - y[1]))

            # 后面一视同仁
            for n in range(2, self.N - 2):
                y_buff2[n] = y_buff1[n] - self.ni * ((1 - self.beta) * (
                        1 * y_buff1[n - 2] - 4 * y_buff1[n - 1] + 6 * y_buff1[n] - 4 * y_buff1[n + 1] + 1 *
                        y_buff1[n + 2]) + self.beta * (y_buff1[n] - y[n]))

        return y_buff2


class Anti_Smooth(nn.Module):
    def __init__(self, N, ni, beta, num_iteration):
        super().__init__()

        self.N = N
        self.ni = ni
        self.beta = beta
        self.num_iteration = num_iteration

    def forward(self, y):

        # 第一轮迭代初始
        # 注意，如果不用.copy()，会直接把变量地址同化，和C++一样
        y_buff1 = y.copy()
        y_buff2 = y.copy()  # 后三个不参与迭代，直接赋予y的值

        y_buff2[0] = y_buff1[0]

        # 先移动特别的y[1]

        y_buff2[1] = y_buff1[1] + self.ni * (
                (1 - self.beta) * (-2 * y_buff1[0] + 5 * y_buff1[1] - 4 * y_buff1[2] + y_buff1[3]) + self.beta * (
                y_buff1[1] - y[1]))

        # 后面一视同仁
        for n in range(2, self.N - 2):
            y_buff2[n] = y_buff1[n] + self.ni * ((1 - self.beta) * (
                    1 * y_buff1[n - 2] - 4 * y_buff1[n - 1] + 6 * y_buff1[n] - 4 * y_buff1[n + 1] + 1 * y_buff1[
                n + 2]) + self.beta * (y_buff1[n] - y[n]))

        # 多轮迭代
        for k in range(0, self.num_iteration):

            y_buff1 = y_buff2.copy()
            y_buff2 = y.copy()

            y_buff2[0] = y_buff1[0]

            # 移动特别的y[1]
            y_buff2[1] = y_buff1[1] + self.ni * ((1 - self.beta) * (
                    -2 * y_buff1[0] + 5 * y_buff1[1] - 4 * y_buff1[2] + y_buff1[3]) + self.beta * (
                                                         y_buff1[1] - y[1]))

            # 后面一视同仁
            for n in range(2, self.N - 2):
                y_buff2[n] = y_buff1[n] + self.ni * ((1 - self.beta) * (
                        1 * y_buff1[n - 2] - 4 * y_buff1[n - 1] + 6 * y_buff1[n] - 4 * y_buff1[n + 1] + 1 *
                        y_buff1[n + 2]) + self.beta * (y_buff1[n] - y[n]))

        return y_buff2


class Random_Smooth(nn.Module):
    def __init__(self, N, ni_smooth, ni_rough, beta, num_iteration):
        super().__init__()

        self.N = N
        self.ni_smooth = ni_smooth
        self.ni_rough = ni_rough
        self.beta = beta
        self.num_iteration = num_iteration

    def forward(self, y):

        # 第一轮迭代初始
        # 注意，如果不用.copy()，会直接把变量地址同化，和C++一样
        y_buff1 = y.copy()
        y_buff2 = y.copy()  # 后三个不参与迭代，直接赋予y的值

        y_buff2[0] = y_buff1[0]

        # 先移动特别的y[1]

        y_buff2[1] = y_buff1[1] + self.ni_smooth * (
                (1 - self.beta) * (-2 * y_buff1[0] + 5 * y_buff1[1] - 4 * y_buff1[2] + y_buff1[3]) + self.beta * (
                y_buff1[1] - y[1]))

        # 后面一视同仁
        for n in range(2, self.N - 2):
            rand = random.getrandbits(1)  # 随机畸变 0-smooth,1-rough
            if rand == 0:
                y_buff2[n] = y_buff1[n] - self.ni_smooth * ((1 - self.beta) * (
                        1 * y_buff1[n - 2] - 4 * y_buff1[n - 1] + 6 * y_buff1[n] - 4 * y_buff1[n + 1] + 1 *
                        y_buff1[n + 2]) + self.beta * (y_buff1[n] - y[n]))
            if rand == 1:
                y_buff2[n] = y_buff1[n] + self.ni_rough * ((1 - self.beta) * (
                        1 * y_buff1[n - 2] - 4 * y_buff1[n - 1] + 6 * y_buff1[n] - 4 * y_buff1[n + 1] + 1 *
                        y_buff1[n + 2]) + self.beta * (y_buff1[n] - y[n]))

        # 多轮迭代
        for k in range(0, self.num_iteration):

            y_buff1 = y_buff2.copy()
            y_buff2 = y.copy()

            y_buff2[0] = y_buff1[0]

            # 移动特别的y[1]
            y_buff2[1] = y_buff1[1] + self.ni_smooth * ((1 - self.beta) * (
                    -2 * y_buff1[0] + 5 * y_buff1[1] - 4 * y_buff1[2] + y_buff1[3]) + self.beta * (
                                                                y_buff1[1] - y[1]))

            # 后面一视同仁
            for n in range(2, self.N - 2):
                rand = random.getrandbits(1)  # 随机畸变 0-smooth,1-rough
                if rand == 0:
                    y_buff2[n] = y_buff1[n] - self.ni_smooth * ((1 - self.beta) * (
                            1 * y_buff1[n - 2] - 4 * y_buff1[n - 1] + 6 * y_buff1[n] - 4 * y_buff1[n + 1] + 1 *
                            y_buff1[n + 2]) + self.beta * (y_buff1[n] - y[n]))
                if rand == 1:
                    y_buff2[n] = y_buff1[n] + self.ni_rough * ((1 - self.beta) * (
                            1 * y_buff1[n - 2] - 4 * y_buff1[n - 1] + 6 * y_buff1[n] - 4 * y_buff1[n + 1] + 1 *
                            y_buff1[n + 2]) + self.beta * (y_buff1[n] - y[n]))
        return y_buff2
