import torch
import torch.nn as nn
from yolo.units import cbl


class SPFF_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernal_size=5) -> None:
        super().__init__()
        c_ = in_channels // 2
        self.cbl1 = cbl.CBL_Unit(in_channels=in_channels,
                                 out_channels=c_,
                                 kernal_size=(1, 1),
                                 stride=(1, 1),
                                 padding='same')

        self.cbl2 = cbl.CBL_Unit(in_channels=c_ * 4,
                                 out_channels=out_channels,
                                 kernal_size=(1, 1),
                                 stride=(1, 1),
                                 padding='same')

        self.maxpool = nn.MaxPool2d(kernel_size=pool_kernal_size,
                                    stride=(1, 1),
                                    padding=pool_kernal_size//2)

    def forward(self, x):
        x = self.cbl1(x)
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        return self.cbl2(torch.concat([x, x1, x2, x3], 1))
