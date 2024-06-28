#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:22:20 2024

@author: jarin.ritu
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy import signal, ndimage
from kornia.geometry.transform import build_laplacian_pyramid
import matplotlib.pyplot as plt
from Utils.Compute_EHD import EHD_Layer
import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # # Compute EHD response for the second pyramid level (x[2])
        # spatial_features = self.ehd_layer(x[2])
        
        # # Get spatial size of second high pass output after EHD
        # spatial_size = spatial_features.shape[-2:]
    
        # Compute EHD response for the first level (x[1])
        features = self.ehd_layer(x[1])
        spatial_size = features.shape[-2:]
        
        # # Resize x[1] to the spatial size of x[2]
        # features = nn.functional.interpolate(features, size=spatial_size, mode="bilinear", align_corners=False)
    
        # Initialize the concatenated features with the resized features for x[1]
        features = [features]
        # features = [features, spatial_features]
    
        # Iterate through the pyramid levels starting from x[2]
        for feature in x[2:]:
            # Compute edge response
            feature = self.ehd_layer(feature)
            
            # Perform fusion (if needed)
            feature = self.weighted_sum(feature)
    
            # Resize feature to the same size as the spatial size from x[2]
            feature = nn.functional.interpolate(feature, size=spatial_size, 
                                                mode="bilinear", align_corners=False)
            features.append(feature)
    
        # Concatenate all features along the channel dimension
        features = torch.cat(features, dim=1)
    
        return features



# # Example usage
# if __name__ == "__main__":
#     # Create a sample image tensor (batch_size=1, channels=1, height, width)
#     image = torch.rand(1, 1, 256, 256)

#     # Initialize the LaplacianPyramidNN
#     laplacian_pyramid_nn = Struct_layer(in_channels=16, max_level=5)

#     # Forward pass to build the pyramid
#     pyramid = laplacian_pyramid_nn(image)

#     # Display the pyramid
#     laplacian_pyramid_nn.display_pyramid(pyramid)
