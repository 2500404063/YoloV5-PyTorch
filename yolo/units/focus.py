import torch
import torch.nn as nn
from yolo.units import cbl


class FOCUS_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, padding='same') -> None:
        super().__init__()
        self.cbl = cbl.CBL_Unit(
            in_channels=in_channels * 4,
            out_channels=out_channels,
            kernal_size=kernal_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x: torch.Tensor):
        if not(x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0):
            raise BaseException("shape must be even number")
        x1 = torch.concat([
            x[..., 0::2, 0::2],
            x[..., 0::2, 1::2],
            x[..., 1::2, 0::2],
            x[..., 1::2, 1::2],
        ], 1)
        return self.cbl(x1)
