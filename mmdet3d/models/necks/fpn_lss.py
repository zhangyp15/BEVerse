# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn import build_norm_layer
from ..builder import NECKS

import pdb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, norm_cfg=dict(type='BN')):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


@NECKS.register_module()
class FPN_LSS(BaseModule):
    def __init__(self, in_channels, out_channels=512, inverse=False, use_neck=True, return_ds=[16, ], init_cfg=None, norm_cfg=dict(type='BN')):
        super(FPN_LSS, self).__init__(init_cfg)
        self.use_neck = use_neck

        if self.use_neck:
            self.up = Up(in_channels, out_channels, norm_cfg=norm_cfg)
            self.inverse = inverse
            self.return_ds = return_ds

    def forward(self, inputs):
        if self.use_neck:
            if self.inverse:
                x2, x1 = inputs
            else:
                x1, x2 = inputs
            if len(self.return_ds) == 1:
                assert self.return_ds[0] == 16
                return self.up(x1, x2)
            else:
                return x1, self.up(x1, x2)
        else:
            return inputs[0]
