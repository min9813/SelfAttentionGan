# %%
import torch
import torch.nn as nn
import numpy as np


class SpectralNorm(nn.Module):

    def __init__(self, module, Ip=1, name="weight", groups=1, bias=True):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.Ip = Ip
        self._make_param()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        w = getattr(self.module, self.name + "_bar")

        height = w.size(0)
        for _ in range(self.Ip):
            v = SpectralNorm.l2_normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = SpectralNorm.l2_normalize(
                torch.mv(w.view(height, -1).data, v))

        sigma = u.dot(w.view(height, -1).mv(v))
        # print("sigma", sigma)

        # print("svd:", torch.svd(w.view(height, -1))[1])

        w = w/sigma.expand_as(w)

        setattr(self.module, self.name, w)
        # u = torch.random.

    def _make_param(self):
        w = getattr(self.module, self.name)

        height = w.size(0)
        _u = nn.Parameter(torch.randn(height))

        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", _u)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, x):
        self._update_u_v()
        return self.module(x)

    @staticmethod
    def l2_normalize(w, eps=1e-12):
        return w/w.norm() + eps


# # %%
# a = nn.Conv2d(1, 3, 2)
# w = a.weight
# print(w)
# print(w.size())
# sn = SpectralNorm(a)
# for k in range(5):
#     sn._update_u_v()
# w = sn.module.weight
# print("check svd:", torch.svd(w.view(w.size(0), -1))[1])
