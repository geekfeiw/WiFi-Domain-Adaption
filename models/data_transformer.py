from .unet_parts import *

class DT(nn.Module):
    def __init__(self, n_channels=150):
        super(DT, self).__init__()

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

        self.out_conv = nn.Conv2d(64, n_channels, kernel_size=3, stride=2, padding=1)
        self.avg = nn.AvgPool2d(kernel_size=16, stride=16, padding=0)


    def forward(self, x):
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
        y = self.out_conv(y)
        y = self.avg(y)
        # y = F.interpolate(y, size=(3, 3), mode='bilinear')

        return y

