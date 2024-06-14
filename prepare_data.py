#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:30:21 2024

@author: jarin.ritu
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division

## PyTorch dependencies
import torch
import pdb

## Local external libraries
from Utils.Get_min_max import get_min_max_minibatch, plot_heatmap,plot_histograms
from Utils.Get_standarize import get_standardization_minibatch
from Datasets.DeepShipSegments import DeepShipSegments
from Datasets.Get_preprocessed_data import process_data


def Prepare_DataLoaders(Network_parameters):
    #pdb.set_trace()
    
    #Data_dirs = {'DeepShip': './Datasets/DeepShip/Segments/'}
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    sample_rate=Network_parameters['sample_rate'][Dataset]
    segment_length=Network_parameters['segment_length'][Dataset]
    #process_data(sample_rate=sample_rate, segment_length=segment_length)

    
    #Change input to network based on models
    #If TDNN or HLTDNN, number of input features is 1
    #Else (CNN), replicate input to be 3 channels
    #If number of input channels is 3 for TDNN, RGB will be set to False
    if (Network_parameters['Model_name'] == 'TDNN' and Network_parameters['TDNN_feats'][Dataset]):
        RGB = False
    else:
        RGB = True
        
    if Dataset == 'DeepShip':
        train_dataset = DeepShipSegments(data_dir, partition='train')
        val_dataset = DeepShipSegments(data_dir, partition='val')
        test_dataset = DeepShipSegments(data_dir, partition='test')        
    else:
        raise RuntimeError('Dataset not implemented') 

    #Compute min max norm of training data for normalization
    #min_value, max_value, norm_function = get_standardization_minibatch(train_dataset, batch_size=128)
    norm_function = get_standardization_minibatch(train_dataset, batch_size=128)
    #plot_heatmap(min_value,max_value,128)
    #plot_histograms(min_value,max_value)
    
    #Set normalization function for each dataset
    train_dataset.norm_function = norm_function
    val_dataset.norm_function = norm_function
    test_dataset.norm_function = norm_function

    #Create dictionary of datasets
    image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    
    # Create training and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=Network_parameters['batch_size'][x],
                                                        shuffle=True,
                                                        num_workers=Network_parameters['num_workers'],
                                                        pin_memory=Network_parameters['pin_memory'])
                                                        for x in ['train', 'val','test']}

    return dataloaders_dict