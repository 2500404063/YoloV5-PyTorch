import torch
import torch.nn as nn
from yolo.units import cbl
from yolo.units import res


class CSP_RES_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, res_amount) -> None:
        super().__init__()
        hidden_out_channels = out_channels // 2
        self.cbl1 = cbl.CBL_Unit(in_channels=in_channels,
                                 out_channels=hidden_out_channels,
                                 kernal_size=kernal_size,
                                 stride=(1, 1),
                                 padding='same')

        self.res_x = nn.ModuleList([res.RES_Unit(in_channels=hidden_out_channels,
                                   out_channels=hidden_out_channels,
                                   kernal_size=kernal_size) for i in range(res_amount)])

        self.conv1 = nn.Conv2d(in_channels=hidden_out_channels,
                               out_channels=hidden_out_channels,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               padding='same')

        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=hidden_out_channels,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               padding='same')

        self.bn = nn.BatchNorm2d(out_channels)

        self.lr = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x1 = self.cbl1(x)
        for res in self.res_x:
            x1 = res(x1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x)
        x2 = torch.concat([x1, x2], 1)
        x2 = self.bn(x2)
        x2 = self.lr(x2)
        return x2
