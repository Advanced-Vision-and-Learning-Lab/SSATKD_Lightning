# -*- coding: utf-8 -*-
"""
Created on Thursday April 25 22:32:00 2024
Wrap models in a PyTorch Lightning Module for training and evaluation
@author: salimalkharsa, jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy
import torch.nn as nn
import visualkeras
## PyTorch dependencies
import torch
import torch.nn as nn
from torchvision import models
## PyTorch dependencies
from torch.nn import functional as F
from Utils.SSTKAD_v2 import SSTKAD
from Utils.CDM_DBF import CustomCDMLayer
from Utils.DTIEM_Model_RBF import QCO_2d
from Utils.CDM import CDM
from torch.profiler import profile, record_function, ProfilerActivity
## Hugging Face dependencies
import timm

## Local external libraries
from Utils.custom_models import Simple_CNN
from Utils.Histogram_Model import HistRes
from Utils.Feature_Extraction_Layer import Feature_Extraction_Layer
from Utils.PANN_models import Cnn14,ResNet38,MobileNetV1,Res1dNet31,Wavegram_Logmel_Cnn14,Cnn14_8k
from Utils.TIMMKD import TIMMKD
import pdb
from torchsummary import summary
from torchviz import make_dot
#from Utils.Focalnet import focalnet_tiny_srf
import os
from Utils.pycontourlet.pycontourlet4d.pycontourlet_module import Pycontourlet
from Utils.struct_layer import Struct_layer
from Utils.EDM import EDM

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def forward_hook(module, input, output):
    print(f"Input shape: {input[0].shape}")
    print(f"Output shape: {output.shape}")

def set_parameter_requires_grad_timm(model):
    
    '''
    This function only works for TIMM models. This will allow for freezing of the backbone and 
    only train the classifier (output layer)
    '''
    #Get classifier name
    for name, param in model.named_parameters():
        if model.default_cfg['classifier'] not in name: 
            param.requires_grad = False
            
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(mode,student_model,teacher_model, in_channels, out_channels, use_pretrained=False, num_classes = 1, feature_extract=False,
                      channels = 3,histogram=True, histogram_layer=None,parallel=True, add_bn=True, scale=5,
                      feat_map_size=4, TDNN_feats=1,window_length=250, 
                      hop_length=64, input_feature='STFT', sample_rate=16000, RGB=False,
                      fusion='all'):
    
    feature_layer = Feature_Extraction_Layer(input_feature=input_feature, window_length=window_length,  window_size=512, hop_size=160, 
        mel_bins=64, fmin=50, fmax=8000, classes_num=527,
        hop_length=hop_length, sample_rate=sample_rate, RGB = RGB )
    

    if mode == 'student' or mode == 'distillation':
        teacher_model_ft = None
        if student_model == "TDNN":
            model_ft = HistRes(histogram_layer, parallel=parallel,
                               model_name=student_model, add_bn=add_bn, scale=scale,
                               pretrained=use_pretrained, TDNN_feats=TDNN_feats)
            set_parameter_requires_grad(model_ft.backbone, feature_extract)

            reduced_dim = int((out_channels / feat_map_size) / (histogram_layer.numBins))
            print('reduced_dim: ', reduced_dim)
        
            if in_channels == reduced_dim:
                model_ft.histogram_layer = histogram_layer
            else:
                conv_reduce = nn.Conv2d(in_channels, reduced_dim, (1, 1))
                model_ft.histogram_layer = nn.Sequential(conv_reduce, histogram_layer)
        
            if parallel:
                num_ftrs = model_ft.fc.in_features * 2
            else:
                num_ftrs = model_ft.fc.in_features

            model_ft.fc = nn.Linear(num_ftrs, num_classes)


    if mode == 'teacher' or mode == 'distillation':
        # model_ft =  None
        if teacher_model == "CNN_14":
            if use_pretrained:
                # Initialize the teacher model
                teacher_model_ft = Cnn14(sample_rate=16000, window_size=250, hop_size=64, mel_bins=64, fmin=50, 
                                         fmax=8000, classes_num=527)
                
                try:
                    # Load the state dictionary of the teacher model
                    teacher_state_dict = torch.load('./PANN_Weights/Cnn14_16k_mAP=0.438.pth')['model']
                    teacher_model_ft.load_state_dict(teacher_state_dict, strict=False)
                except RuntimeError as e:
                    print(f"Ignoring missing keys during loading: {e}")
            else:
                teacher_model_ft = Cnn14(sample_rate=16000, window_size=250, hop_size=64, mel_bins=64, fmin=50, 
                                         fmax=8000, classes_num=527)
                
            #Create feature extaction layer for PANN networks 
            feature_layer.bn = nn.BatchNorm2d(teacher_model_ft.bn0.num_features)

        elif teacher_model == 'ResNet38':
            # feature parameters from repo
            if use_pretrained:  # Pretrained on AudioSet
                teacher_model_ft = ResNet38(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
                    fmax=8000, classes_num=527)
                
                try:
                    teacher_state_dict = torch.load('./PANN_Weights/ResNet38_mAP=0.434.pth')['model']
                    teacher_model_ft.load_state_dict(teacher_state_dict, strict=False)
                except RuntimeError as e:
                    print(f"Ignoring missing key during loading: {e}")
            else:
                teacher_model_ft = ResNet38(sample_rate=32000, window_size=1024, hop_size=160, mel_bins=64, fmin=0, 
                    fmax=None, classes_num=num_classes)
                
            #Create feature extaction layer for PANN networks 
            feature_layer.bn = nn.BatchNorm2d(teacher_model_ft.bn0.num_features)
        
        elif teacher_model == 'MobileNetV1':
            #feature parameters from repo
            if use_pretrained: #Pretrained on AudioSet
                teacher_model_ft = MobileNetV1(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
                    fmax=8000, classes_num=527)
                
                try:
                    teacher_state_dict = torch.load('./PANN_Weights/MobileNetV1_mAP=0.389.pth')['model']
                    teacher_model_ft.load_state_dict(teacher_state_dict, strict=False)
                except RuntimeError as e:
                    print(f"Ignoring missing key during loading: {e}")
            else:
                teacher_model_ft = MobileNetV1(sample_rate=32000, window_size=1024, hop_size=160, mel_bins=64, fmin=0, 
                    fmax=None, classes_num=num_classes)
        
            #Create feature extaction layer for PANN networks 
            feature_layer.bn = nn.BatchNorm2d(teacher_model_ft.bn0.num_features)  
          
         #timm models: convnextv2_base, resnet101,    
        elif teacher_model in timm.list_models():
              teacher_model_ft = timm.create_model(model_name = teacher_model, pretrained = use_pretrained,
                                          num_classes=num_classes,in_chans=channels)        
              #If feature extraction only, freeze backbone and only train output layer
              if feature_extract:
                  num_params = sum(p.numel() for p in teacher_model_ft.parameters() if p.requires_grad)
                  print("Number of parameters before feature extraction: %d" % (num_params))
                  set_parameter_requires_grad_timm(teacher_model_ft)
                  num_params = sum(p.numel() for p in teacher_model_ft.parameters() if p.requires_grad)
                  print("Number of parameters after feature extraction: %d" % (num_params))
                
            #Create custom model to extract 2nd feature maps and output from FC layer
            #Only needed for KD
              teacher_model_ft = TIMMKD(teacher_model_ft,feature_layer)
                
        if teacher_model in timm.list_models(): 
            pass
        else:
            num_ftrs_t = teacher_model_ft.fc_audioset.in_features
            teacher_model_ft.fc_audioset = nn.Linear(num_ftrs_t,num_classes)
            
    # pdb.set_trace()
    
    # filter_op = Pycontourlet()            
            
    # struct_layer =  CDM(in_channels=16)
    # struct_layer = Struct_layer(in_channels=16, max_level=2)
    struct_layer = EDM(in_channels=16,max_level=3,fusion=fusion)
    stats_layer = QCO_2d(scale=1, level_num=8)

    SSTKAD_Model = SSTKAD(feature_layer, model_ft, teacher_model_ft, struct_layer, stats_layer)

    #Return baseline model, desired input size, and feature layer
    return SSTKAD_Model
