#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:57:50 2024
Code modified from https://github.com/xKHUNx/Contourlet-CNN
Takes input feature maps and extracts structural textures at multiple levels using modified ContourletCNN
@author: jarin.ritu
"""

import torch.nn as nn
import torch
import pdb 
from Utils.contourlet_cnn import ContourletCNN
from Utils.pycontourlet.pycontourlet4d.pycontourlet_module import Pycontourlet


class CDM(nn.Module):
    def __init__(self,in_channels, n_levs=[0,3,3,3]):  
        super(CDM, self).__init__()
        
        # Parameters
        self.Pycontourlet = Pycontourlet
        self.n_levs = n_levs
        self.struct_model = ContourletCNN(in_channels=in_channels, 
                                          n_levs=self.n_levs, 
                                          variant="SSFF",
                                          spec_type="all")
        self.out_channels = in_channels

    def forward(self, x):
        pdb.set_trace()
        x = self.struct_model(x)
        
        #Get size of first feature maps
        spatial_size = x[0].shape[-2:]
        
        #Upsample each feature map and concatenate results
        features = nn.functional.interpolate(x[0], size=spatial_size, 
                                            mode="bilinear", align_corners=False)
        
        for index, feature in x.items():
            if index == 0:
                pass
            else:
                #Resize feature to same size as scale 1
                feature = nn.functional.interpolate(feature, size=spatial_size, 
                                                    mode="bilinear", align_corners=False)
                features = torch.cat([features, feature],dim=1)
    
        return features