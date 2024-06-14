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

## PyTorch dependencies
from torch.nn import functional as F

## Hugging Face dependencies
import timm

## Local external libraries
from Utils.custom_models import Simple_CNN
import pdb
#from Utils.Focalnet import focalnet_tiny_srf
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def set_parameter_requires_grad(model):
    
    '''
    This function only works for TIMM models. This will allow for freezing of the backbone and 
    only train the classifier (output layer)
    '''
    #Get classifier name
    for name, param in model.named_parameters():
        if model.default_cfg['classifier'] not in name: 
            param.requires_grad = False
   
def initialize_model(model_name, use_pretrained=False, num_classes = 1, feature_extract=False,
                     # The parameters below are model specific to custom model implementations
                     channels = 3,
                     R=1, measure='norm', p=2.0, similarity=True):
    
    # Initialize these variables which will be set in this if statement. Each of these
    model_ft = None
    input_size = 0

    # If the model is a standard architecture found in timm
    if model_name in timm.list_models():
         model_ft = timm.create_model(model_name = model_name, pretrained = use_pretrained,
                                      num_classes=num_classes,in_chans=channels)
         
         #If feature extraction only, freeze backbone and only train output layer
         if feature_extract:
             num_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
             print("Number of parameters before feature extraction: %d" % (num_params))
             set_parameter_requires_grad(model_ft)
             num_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
             print("Number of parameters after feature extraction: %d" % (num_params))
         input_size = model_ft.default_cfg['input_size'][1]
         
    # If the model is a custom model implementation
    # In this example the custom model is a simple 3 layer CNN
    elif model_name == 'simple_cnn':
        model_ft = Simple_CNN(channels, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

