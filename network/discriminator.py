import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.autograd as autograd
import math


# https://mp.weixin.qq.com/s/dB2M7YT7twipF8up2MZcvA
class ESSAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        B, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)
        attn = t1 + t2
        attn = self.ln(attn)
        x = attn.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


# 实现mixstyle https://github.com/KaiyangZhou/mixstyle-release 即插即用
# 声明：self.mixstyle = MixStyle(p=0.5, alpha=0.1)
class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix


class Discriminator(nn.Module):

    def __init__(self, inchannel, outchannel, num_classes, patch_size):
        super(Discriminator, self).__init__()
        dim = 512
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):  # 前向传播默认test模式

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj


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


class Discriminator_AddDim(nn.Module):  # 针对BEST2-V2,增加可调dim

    def __init__(self, inchannel, outchannel, dim, num_classes, patch_size):
        super(Discriminator_AddDim, self).__init__()
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):  # 前向传播默认test模式

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj


class Discriminator_AddDim_Grad(nn.Module):  # 补充梯度计算

    def __init__(self, inchannel, outchannel, dim, num_classes, patch_size):
        super(Discriminator_AddDim_Grad, self).__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x_y, mode='test'):  # 前向传播默认test模式

        if mode == 'test':

            in_size = x_y.size(0)
            out1 = self.mp(self.relu1(self.conv1(x_y)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            out3 = self.relu3(self.fc1(out2))
            out4 = self.relu4(self.fc2(out3))

            clss = self.cls_head_src(out4)
            return clss

        elif mode == 'train':

            in_size = x_y.size(0)
            x_expand = x_y[:, 0:self.patch_size * self.patch_size * self.inchannel]
            x = x_expand.view(in_size, self.inchannel, self.patch_size, self.patch_size)
            y = x_y[:, -1]

            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            out3 = self.relu3(self.fc1(out2))
            out4 = self.relu4(self.fc2(out3))

            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            y_one_hot = torch.nn.functional.one_hot(y.long(), num_classes=self.num_classes)

            # CE = (clss * y_one_hot).sum()/(self.dim*self.num_classes)
            # CE_grad = autograd.grad(CE, self.cls_head_src.weight, retain_graph=True)[0]

            CE = (clss * y_one_hot).sum()
            CE_grad = autograd.grad(CE, self.conv2.weight, retain_graph=True)[0]

            return clss, proj, CE_grad


class Discriminator_AddDim_ESSAAttn(nn.Module):  # 针对BEST2-V2,增加可调dim

    def __init__(self, inchannel, outchannel, dim, num_classes, patch_size):
        super(Discriminator_AddDim_ESSAAttn, self).__init__()
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.ESSA = ESSAttn(inchannel)
        self.FFN1 = nn.Conv2d(inchannel*2, 64, kernel_size=1, stride=1, padding=0)
        self.leaky = nn.LeakyReLU()
        self.FFN2 = nn.Conv2d(64, inchannel, kernel_size=1, stride=1, padding=0)

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):  # 前向传播默认test模式

        in_size = x.size(0)

        x_ESSA = self.ESSA(x)
        x_ESSA_Cat = torch.cat((x, x_ESSA), 1)
        x_ESSA_En = self.FFN2(self.leaky(self.FFN1(x_ESSA_Cat))) + x

        out1 = self.mp(self.relu1(self.conv1(x_ESSA_En)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj


class Discriminator_CBM(nn.Module):  # SSECNet鉴别器

    def __init__(self, inchannel, outchannel, dim, patch_size, num_classes):
        super(Discriminator_CBM, self).__init__()
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.PReLU()
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.PReLU()
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.PReLU()

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):
        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj


class Discriminator_AddDim_noMP(nn.Module):

    def __init__(self, inchannel, outchannel, dim, num_classes, patch_size):
        super(Discriminator_AddDim_noMP, self).__init__()
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        # self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.relu1(self.conv1(x))
            out2 = self.relu2(self.conv2(out1))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):  # 前向传播默认test模式

        in_size = x.size(0)
        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.conv2(out1))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj


class Discriminator_Twobrach(nn.Module):  # 针对BEST2-V2,增加可调dim

    def __init__(self, inchannel, outchannel, dim, num_classes, patch_size):
        super(Discriminator_Twobrach, self).__init__()
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

        self.conv1_1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu5 = nn.ReLU(inplace=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.mp_1 = nn.MaxPool2d(2)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):  # 前向传播默认test模式

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))

        out1_1 = self.mp_1(self.relu5(self.conv1_1(x)))
        out2_1 = self.mp_1(self.relu6(self.conv2(out1_1)))

        out2 = out2 * torch.sigmoid(out2_1)

        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj


class Discriminator_CBM(nn.Module):  # 针对BEST2-V2,增加可调dim

    def __init__(self, inchannel, outchannel, dim, num_classes, patch_size):
        super(Discriminator_CBM, self).__init__()
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):  # 前向传播默认test模式

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.bn1(self.conv1(x))))
        out2 = self.mp(self.relu2(self.bn2(self.conv2(out1))))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj


class Discriminator_AddDim_Mixstyle(nn.Module):

    def __init__(self, inchannel, outchannel, dim, num_classes, patch_size):
        super(Discriminator_AddDim_Mixstyle, self).__init__()
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

        self.mixstyle = MixStyle(p=0.5, alpha=0.1)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):  # 前向传播默认test模式

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out1 = self.mixstyle(out1)
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj


class Discriminator_AddINTR(nn.Module):

    def __init__(self, inchannel, outchannel, dim, num_classes, patch_size, dim1, dim2):
        super(Discriminator_AddINTR, self).__init__()

        # =============== 前置INTR ===============

        self.patch_size = patch_size

        # for R
        self.conv1 = nn.Conv2d(inchannel, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(dim1, dim2, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)

        self.d_conv2 = nn.ConvTranspose2d(dim2, dim1, kernel_size=3, stride=1, padding=0)
        self.d_conv1 = nn.ConvTranspose2d(dim1, inchannel, kernel_size=5, stride=1, padding=0)

        self.outR = nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=1, padding=0)

        # for S
        self.conv3 = nn.Conv2d(inchannel, dim1, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(dim1, dim2, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)

        self.d_conv4 = nn.ConvTranspose2d(dim2, dim1, kernel_size=3, stride=1, padding=0)
        self.d_conv3 = nn.ConvTranspose2d(dim1, inchannel, kernel_size=5, stride=1, padding=0)

        self.outS = nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=1, padding=0)

        # for  straighten， out_channel有几个，特征约束就需要写几个
        self.S_conv = nn.ConvTranspose2d(dim1, dim1, kernel_size=1, stride=1, padding=0)

        # =====特征融合=====
        self.fusion = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1, stride=1, padding=0)

        # ================== 后置鉴别器 =======================

        self.inchannel = inchannel
        self.conv1_D = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp_D = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存
        self.conv2_D = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp_D(self.relu1(self.conv1_D(x)))
            out2 = self.mp_D(self.relu2(self.conv2_D(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):  # 前向传播默认test模式

        in_size = x.size(0)

        # --------------R---------------------
        r = self.relu(self.bn1(self.conv1(x)))
        r = self.relu(self.bn2(self.conv2(r)))

        f_r = self.S_conv(r)
        f_r_v = f_r.view(in_size, -1)

        r = self.d_conv1(self.d_conv2(r))
        x_r = torch.sigmoid(self.outR(r))

        # ------------- S --------------------
        s = self.relu(self.bn3(self.conv3(x)))
        s = self.relu(self.bn4(self.conv4(s)))

        f_s = self.S_conv(s)
        f_s_v = f_s.view(in_size, -1)

        s = self.d_conv3(self.d_conv4(s))
        x_s = self.outS(s)

        # -------融合处理-------
        x_fusion = self.fusion(torch.cat((x_r, x), 1))

        # -------------D----------------
        out1 = self.mp_D(self.relu1(self.conv1_D(x_fusion)))
        out2 = self.mp_D(self.relu2(self.conv2_D(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            clss = self.cls_head_src(out4)
            return clss, f_r_v, f_s_v, x_r, x_s
