# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:52:40 2020
Function to generate EHD histogram feature maps
@author: jpeeples, luke saleh
"""
import numpy as np
from scipy import signal,ndimage
import torch.nn.functional as F
import pdb
import torch
import torch.types
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
# from skimage.feature import local_binary_pattern
# from torchvision import transforms
# import dask
# import warnings
    
class EHD_Layer(nn.Module):
    def __init__(self, in_channels, angle_res, normalize_kernel,
                 dilation, threshold, window_size, stride, normalize_count,
                aggregation_type, kernel_size= 3,device = 'cpu'):
        
        super(EHD_Layer, self).__init__()
        self.kernel_size = kernel_size
        self.angle_res = angle_res
        self.normalize_kernel = normalize_kernel
        self.in_channels = in_channels
    
        self.dilation = dilation
        self.threshold = threshold
        self.window_size = window_size
        self.window_scale = np.prod(np.asarray(self.window_size))
        self.stride = stride
        self.normalize_count = normalize_count
        self.aggregation_type = aggregation_type

        self.device = device
    
        
        
        #Generate masks based on parameters
        masks = EHD_Layer.Generate_masks(mask_size= self.kernel_size,
                            angle_res= self.angle_res,
                            normalize= self.normalize_kernel)
    
        #Convolve input with filters, expand masks to match input channels
        masks = torch.tensor(masks).float()
        masks = masks.unsqueeze(1)
        
        #Replicate masks along first dimension (for independently)
        masks = masks.repeat(in_channels,1,1,1)

        self.masks = masks
        
        # Call masks now that they are made
        self.num_orientations = self.masks.shape[0] // in_channels
    
    def forward(self,x):
        # pdb.set_trace()
        self.masks = self.masks.to(x.device)
        
        #Treat independently
        x = F.conv2d(x, self.masks,dilation=self.dilation, groups=self.in_channels)
        
        
        #Find max response
        if self.aggregation_type == 'max':
            [value,index] = torch.max(x,dim=1)
            value = value.unsqueeze(1)

        else:
            return x
        
        #Set edge responses to "no edge" if not larger than threshold
        # num_orientations = self.num_orientations
        
        # index[value< self.threshold] = self.num_orientations
        
        return value
    
    @staticmethod
    def Generate_masks(mask_size=3,angle_res=45,normalize=False,rotate=False):
        
        #Make sure masks are appropiate size. Should not be less than 3x3 and needs
        #to be odd size
        if type(mask_size) is list:
            mask_size = mask_size[0]
        if mask_size < 3:
            mask_size = 3
        elif ((mask_size % 2) == 0):
            mask_size = mask_size + 1
        else:
            pass
        
        if mask_size == 3:
            if rotate:
                Gy = np.outer(np.array([1,2,1]).T,np.array([1,0,-1]))
            else:
                Gy = np.outer(np.array([1,0,-1]).T,np.array([1,2,1]))
        else:
            if rotate:
                Gy = np.outer(np.array([1,2,1]).T,np.array([1,0,-1]))
            else:
                Gy = np.outer(np.array([1,0,-1]).T,np.array([1,2,1]))
            dim = np.arange(5,mask_size+1,2)
            expand_mask =  np.outer(np.array([1,2,1]).T,np.array([1,2,1]))
            for size in dim:
                Gy = signal.convolve2d(expand_mask,Gy)
        
        #Generate horizontal masks
        angles = np.arange(0,360,angle_res)
        masks = np.zeros((len(angles),mask_size,mask_size))
        
        #TBD: improve for masks sizes larger than 
        for rot_angle in range(0,len(angles)):
            masks[rot_angle,:,:] = ndimage.rotate(Gy,angles[rot_angle],reshape=False,
                                                mode='nearest')
            
        
        #Normalize masks if desired
        if normalize:
            if mask_size == 3:
                masks = (1/8) * masks
            else:
                masks = (1/8) * (1/16)**len(dim) * masks 
        return masks
    def visualize_feature_maps(self):
        if self.feature_maps is None:
            print("No feature maps to visualize.")
            return
        
        feature_maps = self.feature_maps.squeeze(0)  # Remove batch dimension

        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()

        for idx in range(8):
            axes[idx].imshow(feature_maps[idx].cpu().numpy(), cmap='gray')
            axes[idx].set_title(f'Direction {idx * 45}°')
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()