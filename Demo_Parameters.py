# -*- coding: utf-8 -*-
"""
Created on Thursday April 25 22:32:00 2024
Parameters for training and evaluation of models
@author: jpeeples
"""
import os
import sys

def Parameters(args):
    
    ######## ONLY CHANGE PARAMETERS BELOW ########
    #Flag for if results are to be saved out
    #Set to True to save results out and False to not save results
    save_results = args.save_results
    
    #Location to store trained models
    folder = args.folder
    
    #Select dataset
    data_selection = args.data_selection
    Dataset_names = {1:'FashionMNIST', 2:'CIFAR10', 3:'sugarcane_damage_usa'}
    
    #Flag for feature extraction. False, train whole model. True, only update
    #Flag to use pretrained model from ImageNet or train from scratch (default: True)
    feature_extraction = args.feature_extraction
    use_pretrained = args.use_pretrained
    
    #Set learning rate for new and pretrained (pt) layers
    lr = args.lr
    
    #For no padding, set 0. If padding is desired,
    #enter amount of zero padding to add to each side of image
    #(did not use padding in paper, recommended value is 0 for padding)
    padding = 0
    
    
    #Set step_size and decay rate for scheduler
    #In paper, learning rate was decayed factor of .1 every ten epochs (recommended)
    step_size = 10
    gamma = .1
    
    #Batch size for training and epochs. If running experiments on single GPU (e.g., 2080ti),
    #training batch size is recommended to be 64. If using at least two GPUs,
    #the recommended training batch size is 128 (as done in paper)
    #May need to reduce batch size if CUDA out of memory issue occurs
    batch_size = {'train': args.train_batch_size, 'val': args.val_batch_size, 'test': args.test_batch_size}
    num_epochs = args.num_epochs
    patience = args.patience
    
    #Resize the image before center crop. Recommended values for resize is 256 (used in paper), 384,
    #and 512 (from http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf)
    #Center crop size is recommended to be 256.
    resize_size = args.resize_size
    center_size = 224
    
    #Pin memory for dataloader (set to True for experiments)
    pin_memory = True
    
    #Set number of workers, i.e., how many subprocesses to use for data loading.
    #Usually set to 0 or 1. Can set to more if multiple machines are used.
    #Number of workers for experiments for two GPUs was three
    num_workers = args.num_workers

    #Visualization of results parameters
    #Visualization parameters for figures
    figsize = (12,12)
    fontsize = 12
    
    ######## ONLY CHANGE PARAMETERS ABOVE ########
    if feature_extraction:
        mode = 'Feature_Extraction'
    else:
        mode = 'Fine_Tuning'
    
    #Location of texture datasets
    Data_dirs = {'FashionMNIST': './Datasets/FashionMNIST',
                 'CIFAR10': './Datasets/CIFAR10',
                 'sugarcane_damage_usa': './Datasets/sugarcane_damage_usa'}
    
    # Class names for each dataset
    #If no names available, add dataset name with 'None' (e.g., Dataset_1: None)
    Class_names = {'FashionMNIST': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                     'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
                   'CIFAR10': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                               'dog', 'frog', 'horse', 'ship', 'truck'],
                   'sugarcane_damage_usa': ['cracked', 'crushed', 'no_buds', 'two_buds', 'single_damaged_buds', 'no_damage']}
    
    #Backbone architecture
    Model_name = args.model
    
    #channels in each dataset
    channels = {'FashionMNIST': 1,
                'CIFAR10': 3,
                'sugarcane_damage_usa': 3}
    
    #Number of classes in each dataset
    num_classes = {'FashionMNIST': 10,
                   'CIFAR10': 10,
                   'sugarcane_damage_usa': 6}
    
    #Number of runs and/or splits for each dataset
    Splits = {'FashionMNIST': 3,
              'CIFAR10': 3,
              'sugarcane_damage_usa': 3}
    
    Dataset = Dataset_names[data_selection]
    data_dir = Data_dirs[Dataset]
    class_names = Class_names[Dataset]
    channels = channels[Dataset]
    
    #Return dictionary of parameters
    Params = {'save_results': save_results,'folder': folder,
            'Dataset': Dataset, 'data_dir': data_dir,
            'num_workers': num_workers, 'mode': mode,
            'lr': lr,'step_size': step_size,'gamma': gamma, 
            'batch_size' : batch_size, 'num_epochs': num_epochs, 
            'resize_size': resize_size, 'center_size': center_size, 
            'padding': padding,'Model_name': Model_name, 'num_classes': num_classes, 
            'Splits': Splits, 'feature_extraction': feature_extraction,
            'use_pretrained': use_pretrained,
            'pin_memory': pin_memory, 
            'figsize': figsize,'fontsize': fontsize,
            'channels': channels, 'class_names': class_names,'patience':patience}
    
    return Params
