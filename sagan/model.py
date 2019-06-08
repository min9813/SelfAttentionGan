
import torch
import numpy as np
import torch.nn as nn
import sys
import pathlib
path = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(path))
from sngan.model import SpectralNorm


# class SelfAttnConv(nn.Module):

#     def __init__(self, in_channels, scale=8, kernel_size=1):
#         super(SelfAttnConv, self).__init__()
#         self.query_conv = nn.Conv2d(
#             in_channels, in_channels//scale, kernel_size=1)
#         self.key_conv = nn.Conv2d(
#             in_channels, in_channels//scale, kernel_size=1)
#         self.value_conv = nn.Conv2d(
#             in_channels, in_channels, kernel_size=1)
#         self.softmax = nn.Softmax(dim=-1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         batchsize, C, width, height = x.size()
#         query = self.query_conv(x).view(
#             batchsize, -1, width*height).permute(0, 2, 1)
#         # query shape:(B, C~, N)
#         key = self.key_conv(x).view(batchsize, -1, width*height)
#         # key shape:(B, N, C~)
#         attn = torch.bmm(query, key)
#         attn = self.softmax(attn)
#         # attn shape:(B, N, N)

#         value = self.value_conv(x).view(batchsize, -1, width*height)
#         # value shape:(B, C, N)
#         value = torch.bmm(value, attn.permute(0,2,1))
#         # value shape:(B, C, N)
#         value = value.view(batchsize, -1, width, height)

#         value = self.gamma * value + x

#         return value, attn

class SelfAttnConv(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttnConv, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out, attention


class Generator(nn.Module):

    def __init__(self, attn_channels, z_dim=100, out_channels=3, image_size=64, conv_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        brock_num = int(np.log2(image_size)) - 3
        mult = 2 ** brock_num

        first_layer = []
        first_layer.append(SpectralNorm(
            nn.ConvTranspose2d(z_dim, conv_dim*mult, 4)))
        first_layer.append(nn.BatchNorm2d(conv_dim*mult))
        first_layer.append(nn.ReLU())
        self.l_first = nn.Sequential(*first_layer)

        curr_channel = conv_dim*mult

        print("add self attention layer in {}, {}".format(*attn_channels))
        brock_idx = 0
        brock = []
        curr_channel = conv_dim*mult//(2**(brock_idx+1))

        brock.append(SpectralNorm(
            nn.ConvTranspose2d(curr_channel*2, curr_channel, 4, 2, 1)))
        brock.append(nn.BatchNorm2d(curr_channel))
        brock.append(nn.ReLU())

        self.l1 = nn.Sequential(*brock)
        brock_idx += 1

        brock = []
        curr_channel = conv_dim*mult//(2**(brock_idx+1))

        brock.append(SpectralNorm(
            nn.ConvTranspose2d(curr_channel*2, curr_channel, 4, 2, 1)))
        brock.append(nn.BatchNorm2d(curr_channel))
        brock.append(nn.ReLU())

        self.l2 = nn.Sequential(*brock)
        self.attn1 = SelfAttnConv(attn_channels[0])
        brock_idx += 1

        brock = []
        curr_channel = conv_dim*mult//(2**(brock_idx+1))

        brock.append(SpectralNorm(
            nn.ConvTranspose2d(curr_channel*2, curr_channel, 4, 2, 1)))
        brock.append(nn.BatchNorm2d(curr_channel))
        brock.append(nn.ReLU())

        self.l3 = nn.Sequential(*brock)
        self.attn2 = SelfAttnConv(attn_channels[1])

        last_layer = []
        last_layer.append(SpectralNorm(
            nn.ConvTranspose2d(curr_channel, out_channels, 4, 2, 1)))
        # last_layer.append(nn.BatchNorm2d(out_channels))
        last_layer.append(nn.Tanh())

        self.l_last = nn.Sequential(*last_layer)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.l_first(h)
        h = self.l1(h)
        h = self.l2(h)
        h, attn1 = self.attn1(h)
        h = self.l3(h)
        h, attn2 = self.attn2(h)
        h = self.l_last(h)

        return h, attn1, attn2

    def make_hidden(self, batchsize):
        # return torch.randn
        return torch.randn(batchsize, self.z_dim)


class Discriminator(nn.Module):

    def __init__(self, attn_channels, in_channels=3, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        brock_num = int(np.log2(image_size)) - 3
        mult = 2 ** brock_num

        first_layer = []
        first_layer.append(SpectralNorm(
            nn.Conv2d(in_channels, conv_dim, 4, 2, 1)))
        # first_layer.append(nn.BatchNorm2d(conv_dim))
        first_layer.append(nn.LeakyReLU(0.1))
        self.l_first = nn.Sequential(*first_layer)

        curr_channel = conv_dim

        print("add self attention layer in {}, {}".format(*attn_channels))
        brock_idx = 1
        brock = []

        brock.append(SpectralNorm(
            nn.Conv2d(curr_channel, curr_channel*2, 4, 2, 1)))
        curr_channel = curr_channel*2

        brock.append(nn.LeakyReLU(0.1))

        self.l1 = nn.Sequential(*brock)
        brock_idx += 1

        brock = []

        brock.append(SpectralNorm(
            nn.Conv2d(curr_channel, curr_channel*2, 4, 2, 1)))
        curr_channel = curr_channel*2
        brock.append(nn.LeakyReLU(0.1))

        self.l2 = nn.Sequential(*brock)
        self.attn1 = SelfAttnConv(attn_channels[0])
        brock_idx += 1

        brock = []

        brock.append(SpectralNorm(
            nn.Conv2d(curr_channel, curr_channel*2, 4, 2, 1)))
        curr_channel = curr_channel*2
        brock.append(nn.LeakyReLU(0.1))

        self.l3 = nn.Sequential(*brock)

        assert curr_channel == conv_dim*mult, print(
            "current channel:", curr_channel, "conv dim:", conv_dim*mult)
        self.attn2 = SelfAttnConv(attn_channels[1])

        last_layer = []
        last_layer.append(SpectralNorm(
            nn.Conv2d(curr_channel, 1, 4)))

        self.l_last = nn.Sequential(*last_layer)

    def forward(self, x):
        h = self.l_first(x)
        h = self.l1(h)
        h = self.l2(h)

        h, attn1 = self.attn1(h)
        h = self.l3(h)
        h, attn2 = self.attn2(h)

        h = self.l_last(h)

        return h, attn1, attn2


if __name__ == "__main__":
    g = Generator((128, 64), conv_dim=64)
    d = Discriminator((256, 512), conv_dim=64)
    a = torch.FloatTensor(g.make_hidden(2))
    a, attn1, attn2 = g(a)
    print("finish g")
    a, d_attn1, d_attn2 = d(a)


# # %%
# import torch
# import numpy as np

# a = torch.tensor(np.arange(12).reshape(2,2,3))
# b = torch.tensor(np.arange(18).reshape(2,3,3))

# c = torch.bmm(a,b)
# print(c, c.size())
# c = torch.bmm(b, a.permute(0,2,1))

# print(c, c.size())
