# -*- coding: utf-8 -*-
"""
Created on Thursday April 25 22:32:00 2024
Train and evaluate models for experiments on datasets
@author: jpeeples, salimalkharsa
"""
import argparse

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import glob

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
from objective import objective


import optuna
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#Turn off plotting
plt.ioff()

def main(Params):

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
    
    for split in range(0, numRuns):
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
            data_dir = process_data(sample_rate=Params['sample_rate'], segment_length=Params['segment_length'])
            data_module = DeepShipDataModule(data_dir,Params['batch_size'],
                                             Params['num_workers'], Params['pin_memory'],
                                             train_split=0.7, val_split=0.10, test_split=0.20)
        else:
            raise ValueError('{} Dataset not found'.format(Dataset))
            
        # Initialize the data loaders
        print("Preparing data loaders...")
        data_module.prepare_data()
        data_module.setup("fit")
        data_module.setup("test")
        train_loader, val_loader, test_loader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()
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
        
        if args.mode == 'teacher':
            #model.remove_PANN_feature_extractor()
            model_ft = Lightning_Wrapper(model.teacher, Params['num_classes'][Dataset], 
                                                 log_dir = filename, label_names=Params['class_names'][Dataset])
        elif args.mode == 'distillation':
            # pdb.set_trace()
            
            #Fine tune teacher on dataset
            teacher_checkpoint_callback = ModelCheckpoint(filename = 'best_model_teacher',mode='max',
                                                  monitor='val_accuracy')
            model_ft = Lightning_Wrapper(model.teacher, Params['num_classes'][Dataset], 
                                                  log_dir = filename, label_names=Params['class_names'])
            
            #Train teacher
            print("Setting up teacher trainer...")
            trainer_teacher = Trainer(callbacks=[EarlyStopping(monitor='val_loss', patience=Params['patience']), teacher_checkpoint_callback,
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
                                                                  strict=True)
            model.teacher = best_teacher.model
        
            # Remove feature extraction layers from PANN/TIMM
            model.remove_PANN_feature_extractor()
            model.remove_TIMM_feature_extractor()
            
            model_ft = Lightning_Wrapper_KD(model, num_classes=Params['num_classes'][Dataset],  
                                         log_dir = filename, label_names=Params['class_names'],
                                         Params=Params,criterion=SSTKAD_Loss())
        elif args.mode == 'student':
            model_ft = Lightning_Wrapper(nn.Sequential(model.feature_extractor,model.student),num_classes=Params['num_classes'][Dataset],  
                                         max_iter=len(train_loader), log_dir = filename, label_names=Params['class_names'])
            # pdb.set_trace()
        else:
            raise RuntimeError('{} not implemented'.format(args.mode))
      
        print("Model Initialized as Lightning Module.")
        

        # Create a checkpoint callback to save best model based on val accuracy
        print("Setting up checkpoint callback...")
        checkpoint_callback = ModelCheckpoint(filename = 'best_model',mode='max',
                                              monitor='val_accuracy')
        print("Checkpoint callback set up.")

        # Print number of trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameters: %d" % (num_params))

        # Train and evaluate
        print("Setting up trainer...")
        trainer = Trainer(callbacks=[EarlyStopping(monitor='val_loss', patience=Params['patience']), checkpoint_callback,TQDMProgressBar(refresh_rate=10)], 
                          max_epochs= Params['num_epochs'], enable_checkpointing = Params['save_results'], 
                          default_root_dir = filename,
                          logger=logger) 
        
        
        print("Trainer set up.")
        
        # Start fitting the model
        print('Training model...')
        trainer.fit(model_ft, train_dataloaders = train_loader, val_dataloaders = val_loader)
        print('Training completed.')
        print('Testing model...')
        test_results = trainer.test(model_ft, dataloaders=test_loader)

        # Log test results manually
        for metric, value in test_results[0].items():
            logger.experiment.add_scalar(f'test_{metric}', value)
        
        del model_ft
        torch.cuda.empty_cache()

        print('**********Run ' + str(split + 1) + ' ' + student_model + ' Finished**********')
    
    #     best_val_accs.append(checkpoint_callback.best_model_score.item())

    # average_val_acc = np.mean(best_val_accs)
    # std_val_acc = np.std(best_val_accs)
    # all_runs_accs.append(best_val_accs)

def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction, help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/demo/', help='Location to save models')
    parser.add_argument('--student_model', type=str, default='TDNN', help='Select baseline model architecture')
    parser.add_argument('--teacher_model', type=str, default='CNN_14', help='Select baseline model architecture')
    parser.add_argument('--histogram', default=True, action=argparse.BooleanOptionalAction, help='Flag to use histogram model or baseline global average pooling (GAP), --no-histogram (GAP) or --histogram')
    parser.add_argument('--data_selection', type=int, default=0, help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('-numBins', type=int, default=16, help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--feature_extraction', default=False, action=argparse.BooleanOptionalAction, help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction, help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=32, help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=32, help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=32, help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256, help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction, help='enables CUDA training')
    parser.add_argument('--audio_feature', type=str, default='Log_Mel_Spectrogram', help='Audio feature for extraction')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Select optimizer')
    parser.add_argument('--patience', type=int, default=25, help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for knowledge distillation')
    parser.add_argument('--mode', type=str, choices=['distillation','student', 'teacher'], default='student', help='Mode to run the script in: student, teacher, distillation (default: distillation)')
    parser.add_argument('--HPRC', default=False, action=argparse.BooleanOptionalAction,
                    help='Flag to run on HPRC (default: False)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)
      