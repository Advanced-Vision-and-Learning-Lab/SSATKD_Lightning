#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:09:56 2024

@author: jarin.ritu
"""


import torch
import torch.nn as nn
import numpy as np
import pdb

class TIMMKD(nn.Module):
    def __init__(self,model,feature_extraction,layer_offset=1):

        # inherit nn.module
        super(TIMMKD, self).__init__()

        # define layer properties
        # histogram bin data
        self.model = model
        self.layer_offset = layer_offset
        self.layer_index = len(self.model.feature_info) - self.layer_offset
        self.feature_extraction = feature_extraction
        self.feats_channels = self.model.feature_info[len(self.model.feature_info) - self.layer_index]['num_chs']
      
        
    def forward(self,x): 
        
        #Extract spectrograms
        x = self.feature_extraction(x)
        
        #Extract convolution maps from layer(s) of interest
        feats = self.model.forward_intermediates(x,indices=self.layer_index,intermediates_only=True)[0]
        
        #Get output from network
        x = self.model(x)
        
        return feats, x
        
    