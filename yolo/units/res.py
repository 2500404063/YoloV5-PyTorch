import torch
import torch.nn as nn
from yolo.units import cbl


class RES_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size) -> None:
        super().__init__()
        self.cbl1 = cbl.CBL_Unit(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernal_size=kernal_size,
                                 stride=(1, 1),
                                 padding='same')

        self.cbl2 = cbl.CBL_Unit(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernal_size=kernal_size,
                                 stride=(1, 1),
                                 padding='same')

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               padding='same')

    def forward(self, x):
        x1 = self.cbl1(x)
        x1 = self.cbl2(x1)
        x2 = self.conv1(x)
        return x1 + x2
