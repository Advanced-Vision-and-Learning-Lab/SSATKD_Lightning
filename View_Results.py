# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:33 2019
Generate results from saved models
@author: jpeeples, salimalkharsa
"""

## Python standard libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from Utils.Lightning_Wrapper import Lightning_Wrapper
from DataModules import FashionMNIST_DataModule, CIFAR10_DataModule, sugarcane_damage_usa_DataModule
import glob

## PyTorch Lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger


## PyTorch dependencies
import torch
import torch.nn as nn
import pdb

## Local external libraries
from Demo_Parameters import Parameters
from Utils.Network_functions import initialize_model
from Utils.Save_Results import generate_filename, aggregate_tensorboard_logs, aggregate_and_visualize_confusion_matrices
from Utils.Generate_TSNE_visual import Generate_TSNE_visual


plt.ioff()

def main(Params):    
    # Get the arguments
    args = parse_args()

    # Name of dataset
    Dataset = Params['Dataset']
    
    # Model(s) to be used
    model_name = Params['Model_name'] 

    # Get the label names
    label_names = Params['class_names']

    # Get the phases to run TSNE on
    phases = args.phases

    # Number of runs and/or splits for dataset
    NumRuns = Params['Splits'][Dataset]

    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]
    
    
    for split in range(0, NumRuns):
        torch.manual_seed(split)
        np.random.seed(split)
        np.random.seed(split)
        torch.cuda.manual_seed(split)
        torch.cuda.manual_seed_all(split)
        torch.manual_seed(split)
        
        # Get a filename for saving results
        print("Initializing / Finding the model path...")   
        sub_dir = generate_filename(Params, split)
        checkpt_path = os.path.join(sub_dir, 'lightning_logs/Training/checkpoints/best_model.ckpt')             
        filename = generate_filename(Params,split)
        print("Model path: ", filename)

        # Set up the logger
        print("Setting up logger...")
        logger = TensorBoardLogger(filename, version = 'Val_Test', default_hp_metric=False)
        
        #Remove past events to conserve memory allocation
        log_dir = '{}{}/{}'.format(logger.save_dir,logger.name,logger.version)
        files = glob.glob('{}/{}'.format(log_dir,'events.out.tfevents.*'))
        
        for f in files:
            os.remove(f)
        print("Logger set up.")

        # Initialize the histogram model for this run
        print("Initializing the model...")
        model_ft, input_size = initialize_model(model_name, channels=Params['channels'] , 
                                                use_pretrained=False, 
                                                num_classes = num_classes) 
        print("Model Initialized.")
        
        # Load model
        print('Initializing model as Lightning Module...')
        model = Lightning_Wrapper.load_from_checkpoint( # TODO: Implement map parameter since this is likely useful
            checkpoint_path=checkpt_path, # TODO: Decide how to deal with multiple versions
            # map_location = 
            hparams_file=os.path.join(sub_dir, 'lightning_logs/Training/checkpoints/hparams.yaml'),
            model = model_ft, num_classes = num_classes, strict=True, logger=logger,
            log_dir = filename, label_names = label_names, stage='test')
        print('Model initialized as Lightning Module...')

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
        val_loader = data_module.val_dataloader()
        data_module.setup("test")
        test_loader = data_module.test_dataloader()
        print("Dataloaders Initialized.")
        
        # Create a checkpoint callback
        print("Setting up checkpoint callback...")
        checkpoint_callback = ModelCheckpoint(filename = 'best_model',mode='max',
                                              monitor='val_accuracy')
        print("Checkpoint callback set up.")

          # Train and evaluate
        print("Setting up trainer...")
        trainer = Trainer(callbacks=[checkpoint_callback], 
                          enable_checkpointing = Params['save_results'], 
                          default_root_dir = filename, logger=logger) # forcing it to use CPU
        print("Trainer set up.")
        
        # Validation
        trainer.validate(model, dataloaders = val_loader)
        
        # Test
        trainer.test(model, dataloaders = test_loader)

        if args.TSNE:
            print('Creating TSNE Visual...')

            # Generate TSNE visual
            print("TSNE Phases: ", phases)

            # Remove the fully connected layer from the loaded Lightning Module model
            print('Removing the fully connected layer...')
            # Assumes the model has an attribute 'fc' for the fully connected layer, this is true for Timm models
            if hasattr(model.model, 'fc'):
                # Replace the fully connected layer with an identity layer
                model.model.fc = nn.Identity()  
            # models that use 'classifier' instead of 'fc', also true for Timm models
            elif hasattr(model.model, 'classifier'):  
                model.model.classifier = nn.Identity()
            else:
                raise AttributeError("The model does not have an 'fc' or 'classifier' layer to remove.")
            print('Fully connected layer removed.')

            # Make a dataloaders dictionary for the TSNE that includes the training, validation, and test dataloaders
            dataloaders_dict = {'train': data_module.train_dataloader(), 
                                'val': val_loader, 
                                'test': test_loader} 
                
            Generate_TSNE_visual(dataloaders_dict, model, sub_dir, label_names, phases)

            print('TSNE Visual created.')
    
        print('**********Run ' +  str(split + 1) + '  Finished**********')
    
    print('Getting aggregated results...')
    sub_dir = os.path.dirname(sub_dir.rstrip('/'))
    
    aggregation_folder = 'Aggregated_Results/'
    aggregate_and_visualize_confusion_matrices(sub_dir, aggregation_folder, 
                                               dataset=Dataset,label_names = label_names,
                                               figsize=Params['figsize'], fontsize=Params['fontsize'])
    aggregate_tensorboard_logs(sub_dir, aggregation_folder,Dataset)
    print('Aggregated results saved...')

def parse_args():
    parser = argparse.ArgumentParser(description='Generate results from experiments')
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
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for dataloader (default: 0)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--model', type=str, default='simple_cnn',
                        help='backbone architecture to use (default: simple_cnn)')
    parser.add_argument('--phases', type=list, default=['test'],
                        help='phases to run TSNE on (default: test only, otherwise train,val,test or any combination of the three)')
    parser.add_argument('--TSNE', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag to run TSNE on the data (default: False)')
    parser.add_argument('--patience', type=int, default=10,
                      help='Number of epochs to stop training based on validation loss (default: 10)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)