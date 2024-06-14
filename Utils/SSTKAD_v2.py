#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:19:49 2024

@author: jarin.ritu
"""
import torch
import torch.nn as nn
import numpy as np
import pdb
import torch.nn.functional as F

class SSTKAD(nn.Module):
    def __init__(self, feature_extractor, student, teacher, struct_layer, stats_layer):
        super(SSTKAD, self).__init__()

        self.student = student
        self.teacher = teacher
        self.feature_extractor = feature_extractor
        self.struct_layer = struct_layer
        self.stats_layer = stats_layer
        
        #Freeze teacher network
        # self.set_parameter_requires_grad()

    
        #TBD, add 1x1 convolution
        #Read channels from second layer of teacher autonomously
        self.feature_reduce = nn.Conv2d(64, self.struct_layer.in_channels,1)
        
    def set_parameter_requires_grad(self):
        
        num_params = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        print('Number of learnable parameters for teacher before freezing: {}'.format(num_params))
        
        #Helper function to freeze teacher network for SSTKAD
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        num_params = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        print('Number of learnable parameters for teacher after freezing: {}'.format(num_params))
            
    def remove_PANN_feature_extractor(self):
        #Remove feature extract layers from PANN after intial fine tuning
        #Remove feature layers from PANNs
        self.teacher.spectrogram_extractor = nn.Identity()
        self.teacher.logmel_extractor = nn.Identity()
        self.teacher.spec_augmenter = nn.Identity()
        self.teacher.bn0 = nn.Identity()
        
        #Freeze teacher network
        self.set_parameter_requires_grad()
        
    def forward(self, x):
        
        #Compute spectrogram features using feature layer
        x = self.feature_extractor(x)
        
        #Compute feature maps and outputs from student and teacher
        feats_student, output_student = self.student(x)
        struct_feats_student = self.struct_layer(feats_student)
        feats_teacher, output_teacher = self.teacher(x)
        
        #Match channels and spatial dimension of student and teacher
        feats_teacher = self.feature_reduce(feats_teacher)
        feats_teacher = nn.functional.interpolate(feats_teacher, size=feats_student.shape[-2:], 
                                                  mode="bilinear", align_corners=False)
        
        
        #Pass feature maps through stats and structural modules
        struct_feats_student = self.struct_layer(feats_student)
        struct_feats_teacher = self.struct_layer(feats_teacher)
        
        stats_feats_student = self.stats_layer(feats_student)
        stats_feats_teacher = self.stats_layer(feats_teacher)
        
        
        return struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher
            
        