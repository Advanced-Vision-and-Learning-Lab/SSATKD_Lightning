#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:22:20 2024

@author: jarin.ritu
"""

import torch
import torch.nn as nn
from kornia.geometry.transform import build_laplacian_pyramid
from Utils.Compute_EHD import EHD_Layer
from kornia.augmentation import PadTo
import pdb
import torch.nn.functional as F

def _pad_for_pyramid(x: torch.Tensor, levels: int):
    # x: (B,C,H,W); ensure H,W are multiples of 2^(levels-1)
    H, W = x.shape[-2], x.shape[-1]
    m = 1 << (levels - 1)          # 2**(levels-1)
    ph = (m - (H % m)) % m
    pw = (m - (W % m)) % m
    if ph or pw:
        # (left, right, top, bottom) -> pad bottom/right only
        x = F.pad(x, (0, pw, 0, ph), mode="constant", value=0.0)
    return x, (ph, pw)


class EDM(nn.Module):
    def __init__(self, in_channels, max_level, border_type='reflect', align_corners=False,
                 fusion='max'):
        
        super(EDM, self).__init__()
        #Ensure max level is reached by adding 1 (e.g., for three levels, need 4 input into function)
        self.in_channels = in_channels
        self.max_level = max_level + 1
        self.border_type = border_type
        self.align_corners = align_corners
        self.fusion = fusion
        
        #Initialize edge filters
        self.ehd_layer = EHD_Layer(in_channels=in_channels, angle_res=45, normalize_kernel=True,
                           dilation=1, threshold=0.1, window_size=(3, 3), stride=1,
                           normalize_count=True, aggregation_type=self.fusion, kernel_size=3)
        
        #Define fusion type for edge responses
        if self.fusion == 'weighted':
            self.weighted_sum = nn.Conv2d(in_channels*self.ehd_layer.num_orientations,
                                          in_channels,groups=self.ehd_layer.num_orientations,kernel_size=3, bias=False)
            self.out_channels = self.in_channels * (self.max_level - 1)
        elif self.fusion == 'max':
            self.weighted_sum = nn.Identity()
            self.out_channels = self.in_channels * (self.max_level - 1)
            
        elif self.fusion=='all':
            self.weighted_sum = nn.Identity()
            self.out_channels = self.in_channels * (self.max_level - 1) * self.ehd_layer.num_orientations
            
        else:
            raise RuntimeError('{} fusion method not implemented'.format(self.fusion))
        
 
    def forward(self, x):
        B,C,H0,W0 = x.shape

        # 1) sanitize & pad for pyramid
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x, (ph, pw) = _pad_for_pyramid(x, self.max_level)

        # 2) build pyramid 
        pyr = build_laplacian_pyramid(
            x,
            max_level=self.max_level,
            border_type=self.border_type,
            align_corners=self.align_corners,
        )

        # 3) EHD per level 
        features = self.ehd_layer(pyr[1])
        ref_h, ref_w = features.shape[-2:]
        out = [features]
        for feat in pyr[2:]:
            f = self.ehd_layer(feat)
            f = self.weighted_sum(f)
            if f.shape[-2:] != (ref_h, ref_w):
                f = F.pad(f, (0, ref_w - f.shape[-1], 0, ref_h - f.shape[-2]))
            out.append(f)

        out = torch.cat(out, dim=1)

        # 4) crop back to original if we need to preserve H0×W0
        if ph or pw:
            out = out[..., :H0, :W0]

        return out
