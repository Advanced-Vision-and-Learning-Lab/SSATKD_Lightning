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
        #Compute laplacian pyramid
        x = build_laplacian_pyramid(
            x, max_level=self.max_level, border_type=self.border_type, align_corners=self.align_corners
        )
        # Compute EHD response for the first level (x[1])
        features = self.ehd_layer(x[1])
        spatial_size = features.shape[-2:]
        
    
        # Initialize the concatenated features with the resized features for x[1]
        features = [features] 
        resize_feats = PadTo((spatial_size[0],spatial_size[1]),pad_mode='constant')
    
        # Iterate through the pyramid levels starting from x[2]
        for feature in x[2:]:
            # Compute edge response
            feature = self.ehd_layer(feature)
            
            # Perform fusion (if needed)
            feature = self.weighted_sum(feature)
    
            # Resize feature to the same size as the spatial size from x[2]
            feature = resize_feats(feature)
            features.append(feature)
    
        # Concatenate all features along the channel dimension
        features = torch.cat(features, dim=1)

        return features
