import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F




# class generation(nn.Module):
#     def __init__(self, in_ch=150, out_ch=150, sc=32):
#         super(generation, self).__init__()
#         self.up = nn.Upsample(scale_factor=sc)
#
#     def forward(self, x):
#         x = self.up(x)
#         return x

# super(de_conv, self).__init__()
#         self.generation = nn.Sequential(
#             # 3->6

# class de_conv(nn.Module):
#     '''(conv => BN => ReLU) * 2'''
#     def __init__(self, in_ch, out_ch):
#         super(de_conv, self).__init__()
#         self.generation = nn.Sequential(
#             # 3->6
#             nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#
#             # 6->12
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#
#             # 12->21
#             nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=2, padding=2),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#
#             # 21->42
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#
#             # 42x42->46x82
#             nn.ConvTranspose2d(out_ch, out_ch, kernel_size=[5, 4], stride=[1, 2], padding=[0, 2]),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         x = self.generation(x)
#         return x



class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x