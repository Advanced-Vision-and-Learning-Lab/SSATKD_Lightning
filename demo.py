# -*- coding: utf-8 -*-
"""
Created on Thursday April 25 22:32:00 2024
Train and evaluate models for experiments on datasets
@author: jpeeples, salimalkharsa
"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import glob

from Demo_Parameters import Parameters
from Utils.Save_Results import generate_filename
from Utils.Lightning_Wrapper import Lightning_Wrapper
from Utils.Network_functions import initialize_model
import os

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning import Trainer
#from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
from DataModules import FashionMNIST_DataModule, CIFAR10_DataModule, sugarcane_damage_usa_DataModule

from datetime import timedelta # test this

os.environ['KMP_DUPLICATE_LIB_OK']='True'


#Turn off plotting
plt.ioff()

def main(Params):
    # If running on HPRC
    if Params['HPRC']:
        torch.set_float32_matmul_precision('medium')

    # Name of dataset
    Dataset = Params['Dataset']
    
    # Model(s) to be used
    model_name = Params['Model_name'] 

    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]
    
    # Number of runs and/or splits for dataset
    numRuns = Params['Splits'][Dataset]

    print('Starting Experiments...')
    
    for split in range(0, numRuns):
        #Set same random seed based on split and fairly compare
        #each approach
        torch.manual_seed(split)
        np.random.seed(split)
        np.random.seed(split)
        torch.cuda.manual_seed(split)
        torch.cuda.manual_seed_all(split)
        torch.manual_seed(split)

        print("Initializing/Finding the model path...")                
        filename = generate_filename(Params,split)
        print("Model path: ", filename)

        # Set up the logger (can use Tensorboard, Comet, or other loggers)
        #Other loggers: https://lightning.ai/docs/pytorch/stable/extensions/logging.html
        print("Setting up logger...")
        logger = TensorBoardLogger(filename, default_hp_metric=False, version = 'Training')
        
        #Remove past events to conserve memory allocation
        log_dir = '{}{}/{}'.format(logger.save_dir,logger.name,logger.version)
        files = glob.glob('{}/{}'.format(log_dir,'events.out.tfevents.*'))
        
        for f in files:
            os.remove(f)
        print("Logger set up.")

        # Initialize the histogram model for this run
        print("Initializing the model...")
        model_ft, input_size = initialize_model(model_name, 
                                                use_pretrained=Params['feature_extraction'],
                                                num_classes = num_classes,
                                                feature_extract=Params['feature_extraction'],
                                                channels=Params['channels']) 
        print("Model Initialized.")
        
        # Wrap model in Lightning Module
        print("Initializing Lightning Module...")
        model_ft = Lightning_Wrapper(model=model_ft, num_classes=Params['num_classes'][Dataset],  
                                     log_dir = filename, label_names=Params['class_names'])
        print("Model Initialized as Lightning Module.")

        # Create training and validation dataloaders
        print("Initializing Datasets and Dataloaders...")
        
        # Decide which data module to use
        if Dataset == 'FashionMNIST':
            data_module = FashionMNIST_DataModule(Params['resize_size'], input_size, Params['data_dir'], Params['batch_size'], Params['num_workers'])
            print('FashionMNIST DataModule Initialized')
        elif Dataset == 'CIFAR10':
            data_module = CIFAR10_DataModule(Params['resize_size'], input_size, Params['data_dir'], Params['batch_size'], Params['num_workers'])
            print('CIFAR10 DataModule Initialized')
        elif Dataset == 'sugarcane_damage_usa':
            data_module = sugarcane_damage_usa_DataModule(Params['resize_size'], input_size, Params['data_dir'], Params['batch_size'], Params['num_workers'])
            print('sugarcane_damage_usa DataModule Initialized')
        else:
            raise ValueError('{} Dataset not found'.format(Dataset))
            
        # Initialize the data loaders
        print("Preparing data loaders...")
        data_module.prepare_data()
        data_module.setup("fit")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        print("Dataloaders Initialized.")

        # Create a checkpoint callback to save best model based on val accuracy
        print("Setting up checkpoint callback...")
        checkpoint_callback = ModelCheckpoint(filename = 'best_model',mode='max',
                                              monitor='val_accuracy')
        print("Checkpoint callback set up.")

        # Print number of trainable parameters (if using ACE/Embeddding, only loss layer has params)
        num_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
        print("Number of parameters: %d" % (num_params))

        # Train and evaluate
        print("Setting up trainer...")
        trainer = Trainer(callbacks=[EarlyStopping(monitor='val_loss', patience=Params['patience']), checkpoint_callback,TQDMProgressBar(refresh_rate=1000)], 
                          max_epochs= Params['num_epochs'], enable_checkpointing = Params['save_results'], 
                          default_root_dir = filename,
                          logger=logger) 
        print("Trainer set up.")
        
        # Start fitting the model
        print('Training model...')
        trainer.fit(model_ft, train_dataloaders = train_loader, val_dataloaders = val_loader)
        print('Training completed.')
        
        del model_ft
        torch.cuda.empty_cache()

        print('**********Run ' + str(split + 1) + ' ' + model_name + ' Finished**********')

def parse_args():
    parser = argparse.ArgumentParser(description='Run Pytorch Lightning template experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments(default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models',
                        help='Location to save models')
    parser.add_argument('--data_selection', type=int, default=1,
                        help='Dataset selection:  1:FashionMNIST, 2:CIFAR10, 3:sugarcane_damage_usa')
    parser.add_argument('--feature_extraction', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected/encoder parameters (default: True)')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader (default: 0)')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--patience', type=int, default=10,
                      help='Number of epochs to stop training based on validation loss (default: 10)')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--model', type=str, default='simple_cnn',
                        help='backbone architecture to use (default: simple_cnn)')
    parser.add_argument('--HPRC', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag to run on HPRC (default: False)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)
      
