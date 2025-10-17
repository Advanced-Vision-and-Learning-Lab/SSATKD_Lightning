#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:19:49 2024

@author: jarin.ritu
"""

import torch.nn as nn
from kornia.augmentation import PadTo
import torch.nn.functional as F
import pdb

class SSTKAD(nn.Module):
    def __init__(self, mode, model_group, feature_extractor, student, teacher, struct_layer, stats_layer):
        super(SSTKAD, self).__init__()

        self.student = student
        self.teacher = teacher
        self.feature_extractor = feature_extractor
        self.struct_layer = struct_layer
        self.stats_layer = stats_layer
        self.model_group = model_group
        self.mode = mode
        
        #Freeze teacher network
        # self.set_parameter_requires_grad()

    

        #Read channels from second layer of student autonomously
        self.feature_reduce = nn.Conv2d(64, 16,1)
        self.relu = nn.ReLU()
        
    def set_parameter_requires_grad(self):
        
        num_params = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        # print('Number of learnable parameters for teacher before freezing: {}'.format(num_params))
        if self.mode == 'distillation':
            #Helper function to freeze teacher network for SSTKAD
            for param in self.teacher.parameters():
                param.requires_grad = False
        else:
            for param in self.teacher.parameters():
                param.requires_grad = True
            
        num_params = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        # print('Number of learnable parameters for teacher after freezing: {}'.format(num_params))
            
    def remove_PANN_feature_extractor(self):
        #Remove feature extract layers from PANN after intial fine tuning
        #Remove feature layers from PANNs
        self.teacher.spectrogram_extractor = nn.Identity()
        self.teacher.logmel_extractor = nn.Identity()
        self.teacher.spec_augmenter = nn.Identity()
        self.teacher.bn0 = nn.Identity()
        
        #Freeze teacher network
        self.set_parameter_requires_grad()
    def remove_PANN_feature_extractor_teacher(self):
        #Remove feature extract layers from PANN after intial fine tuning
        #Remove feature layers from PANNs
        self.teacher.spectrogram_extractor = nn.Identity()
        self.teacher.logmel_extractor = nn.Identity()
        self.teacher.spec_augmenter = nn.Identity()
        self.teacher.bn0 = nn.Identity()
        
        
    def remove_TIMM_feature_extractor(self):
        
        #Remove feature extract layers from PANN after intial fine tuning
        #Remove feature layers from PANNs
        self.teacher.feature_extraction = nn.Identity()
        
        #Freeze teacher network
        self.set_parameter_requires_grad()

    def forward(self, x):
        if self.model_group =='Spectogram':
        # #Compute spectrogram features using feature layer
            x = self.feature_extractor(x)
        
        #Compute feature maps and outputs from student and teacher
        feats_student, output_student = self.student(x)
        feats_teacher, output_teacher = self.teacher(x)
        if self.model_group =='Spectogram':
            #Match channels and spatial dimension of student and teacher
            feats_teacher = self.feature_reduce(feats_teacher)
            feats_teacher = self.relu(feats_teacher)
        
        
        # size = feats_student.shape[-2:]
        # resize_feats_teacher = PadTo((size[0],size[1]),pad_mode='constant')                                                                     
        # feats_teacher = resize_feats_teacher(feats_teacher)
        # --- Spatial align: match both to the smaller H×W (per-dimension) ---
        sH, sW = feats_student.shape[-2], feats_student.shape[-1]
        tH, tW = feats_teacher.shape[-2], feats_teacher.shape[-1]
        
        Ht, Wt = min(sH, tH), min(sW, tW)  # target = smaller per dimension
        
        if (sH, sW) != (Ht, Wt):
            feats_student = F.interpolate(feats_student, size=(Ht, Wt), mode="bilinear", align_corners=False)
        if (tH, tW) != (Ht, Wt):
            feats_teacher = F.interpolate(feats_teacher, size=(Ht, Wt), mode="bilinear", align_corners=False)
                                                  

        
        struct_feats_student = self.struct_layer(feats_student)
        struct_feats_teacher = self.struct_layer(feats_teacher)
        # print("\ struct_feats_teacher",struct_feats_teacher[:2, :2, :2, :2])
        
        stats_feats_student = self.stats_layer(feats_student)
        # print("\n stats",stats_feats_student[:2, :2, :2, :2])
        stats_feats_teacher = self.stats_layer(feats_teacher)

    
        
        return struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher

     
     
