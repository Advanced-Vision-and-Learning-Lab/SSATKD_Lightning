#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:48:24 2024

@author: jarin.ritu
"""

import argparse

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import glob
import optuna

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from Utils.Lightning_Wrapper import Lightning_Wrapper_KD
from DeepShipDataModules import DeepShipDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar


from Demo_Parameters import Parameters
from Utils.Save_Results import generate_filename
from Utils.Lightning_Wrapper import Lightning_Wrapper, Lightning_Wrapper_KD
from Utils.Network_functions import initialize_model
import os

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from DeepShipDataModules import DeepShipDataModule

from Utils.RBFHistogramPooling import HistogramLayer
import pdb
from Datasets.Get_preprocessed_data import process_data
from Utils.Loss_function import SSTKAD_Loss
import csv

def objective(trial, Params):
    stats_w = trial.suggest_float('stats_w', 0.2, 2)
    struct_w = trial.suggest_float('struct_w', 0.2, 2)
    distill_w = trial.suggest_float('distll_w', 0.2, 2)


    # Ensure the sum of weights equals 1
    weight_sum = stats_w + struct_w + distill_w 
    stats_w /= weight_sum
    struct_w /= weight_sum
    distill_w /= weight_sum
   # Existing main function logic with updated Params
    if Params['HPRC']:
        torch.set_float32_matmul_precision('medium')

    # Name of dataset
    Dataset = Params['Dataset']
    
    # Model(s) to be used
    student_model = Params['student_model'] 
    teacher_model = Params['teacher_model']

    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]
    
    # Number of runs and/or splits for dataset
    numRuns = Params['Splits'][Dataset]
    numBins = Params['numBins']
    num_feature_maps = Params['out_channels'][student_model]
    mode = Params['mode']
    
    # Local area of feature map after histogram layer
    feat_map_size = Params['feat_map_size']

    print('Starting Experiments...')
    
    for split in range(0, 1):
        #Set same random seed based on split and fairly compare
        #each approach
        torch.manual_seed(split)
        np.random.seed(split)
        np.random.seed(split)
        torch.cuda.manual_seed(split)
        torch.cuda.manual_seed_all(split)
        torch.manual_seed(split)
        best_val_accs = []
        all_runs_accs = []

        histogram_layer = HistogramLayer(int(num_feature_maps / (feat_map_size * numBins)),
                                         Params['kernel_size'][student_model],
                                         num_bins=numBins, stride=Params['stride'],
                                         normalize_count=Params['normalize_count'],
                                         normalize_bins=Params['normalize_bins'])

        print("Initializing/Finding the model path...")                
        filename = generate_filename(Params,split)
        print("Model path: ", filename)

        # Set up the logger (can use Tensorboard, Comet, or other loggers)
        #Other loggers: https://lightning.ai/docs/pytorch/stable/extensions/logging.html
        print("Setting up logger...")
        logger = TensorBoardLogger(filename, default_hp_metric=False, version = 'Training')
        
        #Remove past events to conserve memory allocation
        log_dir = '{}{}/{}'.format(logger.save_dir,logger.name,logger.version)
        print(log_dir)

        files = glob.glob('{}/{}'.format(log_dir,'events.out.tfevents.*'))
        
        for f in files:
            os.remove(f)
        print("Logger set up.")
        
        # Create training and validation dataloaders
        print("Initializing Datasets and Dataloaders...")
        
        # Decide which data module to use
        #*********Use smaller training percentage for debugging**********
        if Dataset == 'DeepShip':
            # process_data(sample_rate=Params['sample_rate'], segment_length=Params['segment_length'])
            data_module = DeepShipDataModule(Params['data_dir'],Params['batch_size'],
                                             Params['num_workers'], Params['pin_memory'],
                                             train_split=0.7, val_split=0.1, test_split=0.2)
        else:
            raise ValueError('{} Dataset not found'.format(Dataset))
            
        # Initialize the data loaders
        print("Preparing data loaders...")
        data_module.prepare_data()
        data_module.setup("fit")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        print("Dataloaders Initialized.")

        # Initialize the histogram model for this run
        print("Initializing the model...")
        model = initialize_model(mode, student_model, teacher_model, 
                                                Params['in_channels'][student_model], num_feature_maps,
                                                use_pretrained=Params['use_pretrained'],
                                                num_classes = num_classes,
                                                feature_extract=Params['feature_extraction'],
                                                channels=Params['TDNN_feats'][Dataset],
                                                histogram=Params['histogram'],
                                                histogram_layer=histogram_layer,
                                                parallel=Params['parallel'],
                                                add_bn=Params['add_bn'],
                                                scale=Params['scale'],
                                                feat_map_size=feat_map_size,
                                                TDNN_feats=(Params['TDNN_feats'][Dataset]),
                                                window_length=(Params['window_length'][Dataset]), 
                                                hop_length=(Params['hop_length'][Dataset]),
                                                input_feature = Params['feature'],
                                                sample_rate=Params['sample_rate']) 
        
        
        # Wrap model in Lightning Module
        print("Initializing Lightning Module...")
        teacher_checkpoint_callback = ModelCheckpoint(filename = 'best_model_teacher',mode='max',
                                              monitor='val_accuracy')
        model_ft = Lightning_Wrapper(model.teacher, Params['num_classes'][Dataset], 
                                              log_dir = filename, label_names=Params['class_names'])
                     
        #Train teacher
        print("Setting up teacher trainer...")
        trainer_teacher = Trainer(callbacks=[EarlyStopping(monitor='val_loss', patience=Params['patience']), 
                                             teacher_checkpoint_callback,
                                      TQDMProgressBar(refresh_rate=10)], 
                          max_epochs= Params['num_epochs'], enable_checkpointing = Params['save_results'], 
                          default_root_dir = filename,
                          logger=logger) 
        
        
        print("Teacher trainer set up.")
        
        # Start fitting the model
        print('Training teacher model...')
        
        trainer_teacher.fit(model_ft, train_dataloaders = train_loader, 
                            val_dataloaders = val_loader)
        print('Training completed.')
        
        #Pass fine-tuned teacher to knowledge distillation model
        sub_dir = generate_filename(Params, split)
        checkpt_path = os.path.join(sub_dir, 'lightning_logs/Training/checkpoints/best_model_teacher.ckpt')             
        best_teacher = Lightning_Wrapper.load_from_checkpoint(checkpt_path,
                                                              hparams_file=os.path.join(sub_dir, 'lightning_logs/Training/checkpoints/hparams.yaml'),
                                                              model=model.teacher,
                                                              num_classes = num_classes, 
                                                              sample_rate=Params['sample_rate']) 
    
        model.teacher = best_teacher.model
        # Remove feature extraction layers from PANN/TIMM
        model.remove_PANN_feature_extractor()
        model.remove_TIMM_feature_extractor()
            
        model_ft = Lightning_Wrapper_KD(model, num_classes=Params['num_classes'][Dataset],stats_w=stats_w,
                                    struct_w=struct_w,distill_w=distill_w,
                                     log_dir = filename, label_names=Params['class_names'],
                                     Params=Params,criterion=SSTKAD_Loss())

        #model = Lightning_Wrapper_KD(stats_w=weight1, struct_w=weight2, distll_w=weight3)
        print("Model Initialized as Lightning Module.")
        

        # Create a checkpoint callback to save best model based on val accuracy
        print("Setting up checkpoint callback...")
        checkpoint_callback = ModelCheckpoint(filename = 'best_model',mode='max',
                                              monitor='val_accuracy')
        print("Checkpoint callback set up.")

        # Print number of trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameters: %d" % (num_params))
        pdb.set_trace()
        trainer = Trainer(
            max_epochs=100, 
            logger=TensorBoardLogger('tb_logs', name='my_model'),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=Params['patience']),
                ModelCheckpoint(monitor='val_accuracy', mode='max', filename='best_model'),
                TQDMProgressBar(refresh_rate=10)
            ]
        )
        trainer.fit(model_ft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Assuming the validation accuracy is logged as 'val_accuracy'
        val_accuracy = trainer.callback_metrics['val_accuracy'].item()
    
        # Save trial results
        with open('optuna_trial_results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([trial.number, val_accuracy,stats_w,struct_w,distill_w])
        
        return val_accuracy

        

