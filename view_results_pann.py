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
from DeepShipDataModules import DeepShipDataModule
from Utils.Lightning_Wrapper import Lightning_Wrapper
from Utils.Lightning_Wrapper import Lightning_Wrapper, Lightning_Wrapper_KD
import glob

## PyTorch Lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from Utils.RBFHistogramPooling import HistogramLayer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateFinder

## PyTorch dependencies
import torch
import torch.nn as nn
import pdb
from Datasets.Get_preprocessed_data import process_data
## Local external libraries
from Demo_Parameters import Parameters
from Utils.Network_functions import initialize_model
from Utils.Save_Results import generate_filename, aggregate_tensorboard_logs, aggregate_and_visualize_confusion_matrices


plt.ioff()

def main(Params):    
    # Get the arguments
    args = parse_args()
    if params['HPRC']:
        torch.set_float32_matmul_precision('medium')
    # Name of dataset
    Dataset = Params['Dataset']
    student_model = Params['student_model'] 
    teacher_model = Params['teacher_model']

    numRuns = Params['Splits'][Dataset]
    numBins = Params['numBins']
    num_feature_maps = Params['out_channels'][student_model]
    mode = Params['mode']
    feat_map_size = Params['feat_map_size']


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
        
        # pdb.set_trace()
        
        filename = generate_filename(Params, split)
        logger = TensorBoardLogger(
            save_dir=os.path.join(filename, "tb_logs"),
            name="model_logs",
        )
        
        #Remove past events to conserve memory allocation
        log_dir = '{}{}/{}'.format(logger.save_dir,logger.name,logger.version)
        files = glob.glob('{}/{}'.format(log_dir,'events.out.tfevents.*'))
        
        for f in files:
            os.remove(f)
        print("Logger set up.")
        if Dataset == 'DeepShip':
            data_dir = process_data(sample_rate=Params['sample_rate'], segment_length=Params['segment_length'])
            data_module = DeepShipDataModule(
                data_dir, Params['batch_size'],
                Params['num_workers'], Params['pin_memory'],
                train_split=0.70, val_split=0.10, test_split=0.20
            )
        else:
            raise ValueError(f'{Dataset} Dataset not found')
        # Initialize the histogram model for this run
                    #Initialize histogram layer based on type
        histogram_layer = HistogramLayer(
            int(num_feature_maps / (feat_map_size * numBins)),
            Params['kernel_size'][student_model],
            num_bins=numBins, stride=Params['stride'],
            normalize_count=Params['normalize_count'],
            normalize_bins=Params['normalize_bins']
        )
        # Create training and validation dataloaders
        print("Initializing Datasets and Dataloaders...")
        # Decide which data module to use
        
            
        print("Preparing data loaders...")
        data_module.prepare_data()
        data_module.setup("fit")
        data_module.setup(stage='test')
        # pdb.set_trace()
        train_loader, val_loader, test_loader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()
        # pdb.set_trace()

        # print(f"Label distribution in test set: {label_counts}")
        print("Dataloaders Initialized.")
        
        print("Initializing the model...")
        model = initialize_model(
            mode, student_model, teacher_model, 
            Params['in_channels'][student_model], num_feature_maps,
            use_pretrained=Params['use_pretrained'],
            num_classes=num_classes,
            feature_extract=Params['feature_extraction'],
            channels=Params['TDNN_feats'][Dataset],
            histogram=Params['histogram'],
            histogram_layer=histogram_layer,
            parallel=Params['parallel'],
            add_bn=Params['add_bn'],
            scale=Params['scale'],
            feat_map_size=feat_map_size,
            TDNN_feats=Params['TDNN_feats'][Dataset],
            window_length=Params['window_length'][Dataset], 
            hop_length=Params['hop_length'][Dataset],
            input_feature=Params['feature'],
            sample_rate=Params['sample_rate']
        )
        print("Model Initialized.")
        
        # Load model
        print('Initializing model as Lightning Module...')
        
        checkpoint_callback = ModelCheckpoint(filename = 'best_model',mode='max',
                                              monitor='val_accuracy')
        best_model_path = checkpoint_callback.best_model_path  
        model = Lightning_Wrapper_KD.load_from_checkpoint(
            best_model_path,
            hparams_file=os.path.join(filename, 'tb_logs/model_logs/version_0/checkpoints/hparams.yaml'),
            model= model,
            num_classes=num_classes,max_iter=len(train_loader),
            strict=True
        )
        print('Model initialized as Lightning Module...')


        # # Create a checkpoint callback
        # print("Setting up checkpoint callback...")
        # checkpoint_callback = ModelCheckpoint(filename = 'best_model',mode='max',
        #                                       monitor='val_accuracy')
        print("Checkpoint callback set up.")

          # Train and evaluate
        # Initialize the trainer with the custom learning rate finder callback
        trainer = Trainer(callbacks=[EarlyStopping(monitor='val_loss', patience=Params['patience']), checkpoint_callback,TQDMProgressBar(refresh_rate=10)], 
                          max_epochs= Params['num_epochs'], enable_checkpointing = Params['save_results'], 
                          default_root_dir = filename,
                          logger=logger) # forcing it to use CPU
        print("Trainer set up.")
        
        # Validation
        trainer.validate(model, dataloaders = val_loader)
        
        # Test
        trainer.test(model, dataloaders = test_loader)


    
    print('Getting aggregated results...')
    sub_dir = os.path.dirname(filename.rstrip('/'))
    
    aggregation_folder = 'Aggregated_Results/'
    aggregate_and_visualize_confusion_matrices(sub_dir, aggregation_folder, 
                                               dataset=Dataset,label_names = Params['class_names'][Dataset],
                                               figsize=Params['figsize'], fontsize=Params['fontsize'])
    aggregate_tensorboard_logs(sub_dir, aggregation_folder,Dataset)
    print('Aggregated results saved...')

def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction, help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/cm_test', help='Location to save models')
    parser.add_argument('--student_model', type=str, default='TDNN', help='Select baseline model architecture')
    parser.add_argument('--teacher_model', type=str, default='CNN_14', help='Select baseline model architecture')
    parser.add_argument('--histogram', default=True, action=argparse.BooleanOptionalAction, help='Flag to use histogram model or baseline global average pooling (GAP), --no-histogram (GAP) or --histogram')
    parser.add_argument('--data_selection', type=int, default=0, help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('-numBins', type=int, default=16, help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--feature_extraction', default=False, action=argparse.BooleanOptionalAction, help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction, help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=64, help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=128, help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=128, help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256, help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction, help='enables CUDA training')
    parser.add_argument('--audio_feature', type=str, default='STFT', help='Audio feature for extraction')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Select optimizer')
    parser.add_argument('--ablation', type=str, default='True', help='Select ablation study to be true or false')
    parser.add_argument('--patience', type=int, default=50, help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for knowledge distillation')
    parser.add_argument('--mode', type=str, choices=['distillation','student', 'teacher'], default='distillation', help='Mode to run the script in: student, teacher, distillation (default: distillation)')
    parser.add_argument('--HPRC', default=False, action=argparse.BooleanOptionalAction,
                    help='Flag to run on HPRC (default: False)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)