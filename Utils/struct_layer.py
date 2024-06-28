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
import kornia
import matplotlib.pyplot as plt
from Utils.Compute_EHD import EHD_Layer
import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Struct_layer(nn.Module):
    def __init__(self, in_channels, max_level, border_type='reflect', align_corners=False):
        super(Struct_layer, self).__init__()
        self.max_level = max_level
        self.border_type = border_type
        self.align_corners = align_corners
        self.out_channels = in_channels
        self.ehd_layer = EHD_Layer(in_channels=16, angle_res=45, normalize_kernel=True,
                           dilation=1, threshold=0.1, window_size=(3, 3), stride=1,
                           normalize_count=True, aggregation_type='GAP', kernel_size=3)
        
     
        
        
    def forward(self, x):
        # pdb.set_trace()
        pyramid = self.build_laplacian_pyramid(
            x, max_level=self.max_level, border_type=self.border_type, align_corners=self.align_corners
        )
        
        low_pass = pyramid[-1]
        high_pass_bands = pyramid[:-1]
    
        # Process high-pass subbands with EHD layer
        high_pass_processed = [self.ehd_layer(band)[0] for band in high_pass_bands]
    
        # Further decompose the low-pass subband
        further_pyramid = self.build_laplacian_pyramid(
            low_pass, max_level=self.max_level, border_type=self.border_type, align_corners=self.align_corners
        )
    
        # Combine the results
        combined = high_pass_processed + further_pyramid
    
        return combined


    def build_laplacian_pyramid(self, x, max_level, border_type, align_corners):
        pyramid = []
        current = x
        for level in range(max_level):
            if current.size(2) < 2 or current.size(3) < 2:
                break  # Stop if the image is too small to downsample
            next_level = self.pyrdown_safe(current, border_type)
            pyramid.append(current - self.pyrup_safe(next_level, current.size(), border_type))
            current = next_level
        pyramid.append(current)
        pdb.set_trace()
        return pyramid

    def pyrdown_safe(self, x, border_type):
        print(f"Input shape before pyrdown_safe: {x.shape}")
        kernel = torch.tensor([[1., 4., 6., 4., 1.],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]]) / 256.
        kernel = kernel.unsqueeze(0).unsqueeze(0)  
        
        kernel = kernel.repeat(x.size(1), 1, 1, 1).to(device)
        
        padding_shape = (kernel.size(2) // 2, kernel.size(3) // 2)
        padding_shape = (min(padding_shape[0], x.size(2) - 1), min(padding_shape[1], x.size(3) - 1))
        
        x_blur = F.pad(x, (padding_shape[1], padding_shape[1], padding_shape[0], padding_shape[0]), mode=border_type).to(device)
        x_blur = F.conv2d(x_blur, kernel, groups=x.size(1))
        print(f"Output shape after conv2d in pyrdown_safe: {x_blur.shape}")
        
        return x_blur[:, :, ::2, ::2]

    def pyrup_safe(self, x, target_size, border_type):
        x_up = torch.zeros((x.size(0), x.size(1), target_size[2], target_size[3]))
        x_up[:, :, ::2, ::2] = x
        kernel = torch.tensor([[1., 4., 6., 4., 1.],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]]) / 256.
        kernel = kernel.unsqueeze(0).unsqueeze(0) 
        
        kernel = kernel.repeat(x.size(1), 1, 1, 1).to(device)
        padding_shape = (kernel.size(2) // 2, kernel.size(3) // 2)
        padding_shape = (min(padding_shape[0], target_size[2] - 1), min(padding_shape[1], target_size[3] - 1))
        x_up = F.pad(x_up, (padding_shape[1], padding_shape[1], padding_shape[0], padding_shape[0]), mode=border_type).to(device)
        x_up = F.conv2d(x_up, kernel, groups=x.size(1))
        return x_up

    def display_pyramid(self, pyramid):
        num_levels = len(pyramid)
        fig, axes = plt.subplots(1, num_levels, figsize=(15, 5))
        if num_levels == 1:
            axes = [axes]
        
        for i, (ax, tensor) in enumerate(zip(axes, pyramid)):
            ax.imshow(tensor.squeeze().detach().cpu().numpy(), cmap='gray')
            ax.set_title(f'Level {i+1}')
            ax.axis('off')
        
        plt.show()





















#     def forward(self, x):
#         # pdb.set_trace()
#         pyramid = kornia.geometry.transform.build_laplacian_pyramid(
#             x, max_level=self.max_level, border_type=self.border_type, align_corners=self.align_corners
#         )
#         return pyramid

#     def display_pyramid(self, pyramid):
#         num_levels = len(pyramid)
#         fig, axes = plt.subplots(1, num_levels, figsize=(10, 6))
#         for i, level in enumerate(pyramid):
#             img = level[0, 0].cpu().numpy()  # Adjusted for batch and channel dimensions
#             axes[i].imshow(img, cmap='gray')
#             axes[i].set_title(f'Level {i}')
#             axes[i].axis('off')
#         plt.show()



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
