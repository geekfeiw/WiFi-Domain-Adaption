from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels=150, n_heat=15, n_paf=14*2, n_bg=1, n_seg=1):
        super(UNet, self).__init__()

        # self.up = generation(n_channels, n_channels)

        self.inc = inconv(n_channels, 64)

        self.down_heat_1 = down(64, 128)
        self.down_heat_2 = down(128, 256)
        self.down_heat_3 = down(256, 512)
        self.down_heat_4 = down(512, 512)
        self.up_heat_1 = up(1024, 256)
        self.up_heat_2 = up(512, 128)
        self.up_heat_3 = up(256, 64)
        self.up_heat_4 = up(128, 64)
        self.out_heat = outconv(64, 64)

        # 96x96 -> 46x82
        # self.resize_heat = nn.Upsample(size=(94, 84), mode='bilinear')
        # 96x96 -> 46x82
        self.resize_heats = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=[4, 3], stride=[2, 1], padding=[2, 1]),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.hm = nn.Conv2d(32, n_heat, kernel_size=3, stride=1, padding=0)
        self.bg = nn.Conv2d(32, n_bg, kernel_size=3, stride=1, padding=0)

        self.down_paf_1 = down(64, 128)
        self.down_paf_2 = down(128, 256)
        self.down_paf_3 = down(256, 512)
        self.down_paf_4 = down(512, 512)
        self.up_paf_1 = up(1024, 256)
        self.up_paf_2 = up(512, 128)
        self.up_paf_3 = up(256, 64)
        self.up_paf_4 = up(128, 64)
        self.out_paf = outconv(64, 64)

        # self.resize_paf = nn.Upsample(size=(94, 84), mode='bilinear')
        # 96x96 -> 46x82
        self.resize_pafs = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=[4, 3], stride=[2, 1], padding=[2, 1]),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, n_paf, kernel_size=3, stride=1, padding=0),
        )

    def forward(self, x):
        # x = self.up(x)
        x = F.interpolate(x, scale_factor=32, mode='bilinear')

        x1 = self.inc(x)

        y1 = self.down_heat_1(x1)
        y2 = self.down_heat_2(y1)
        y3 = self.down_heat_3(y2)
        y4 = self.down_heat_4(y3)
        y = self.up_heat_1(y4, y3)
        y = self.up_heat_2(y, y2)
        y = self.up_heat_3(y, y1)
        y = self.up_heat_4(y, x1)
        y = self.out_heat(y)
        # y = self.resize_heat(y)
        y = F.interpolate(y, size=(94, 84), mode='bilinear')
        y = self.resize_heats(y)

        keypoints_hm = self.hm(y)
        background_hm = self.bg(y)

        z1 = self.down_paf_1(x1)
        z2 = self.down_paf_2(z1)
        z3 = self.down_paf_3(z2)
        z4 = self.down_paf_4(z3)
        z = self.up_paf_1(z4, z3)
        z = self.up_paf_2(z, z2)
        z = self.up_paf_3(z, z1)
        z = self.up_paf_4(z, x1)
        z = self.out_paf(z)
        # z = self.resize_paf(z)
        z = F.interpolate(z, size=(94, 84), mode='bilinear')
        z = self.resize_pafs(z)

        return keypoints_hm, background_hm, z

