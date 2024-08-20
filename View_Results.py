# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:33 2019
Generate results from saved models
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import os
from sklearn.metrics import matthews_corrcoef
import pickle
import argparse
from lightning import Trainer

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Utils.Generate_TSNE_visual import Generate_TSNE_visual
from Utils.Class_information import Class_names
from Demo_Parameters import Parameters
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateFinder
from Utils.Network_functions import initialize_model
from Utils.RBFHistogramPooling import HistogramLayer
# from Utils.Confusion_mats import plot_confusion_matrix, plot_avg_confusion_matrix
# from Utils.Generate_Learning_Curves import Plot_Learning_Curves
from Utils.Save_Results import generate_filename
from Datasets.Get_preprocessed_data import process_data
from DeepShipDataModules import DeepShipDataModule
from Utils.Lightning_Wrapper import Lightning_Wrapper, Lightning_Wrapper_KD
from Utils.Loss_function import SSTKAD_Loss
import pdb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from Utils.Save_Results import generate_filename, aggregate_tensorboard_logs, aggregate_and_visualize_confusion_matrices


import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.ioff()



def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
def main(Params):

    torch.set_float32_matmul_precision('medium')
    fig_size = Params['fig_size']
    font_size = Params['font_size']
    
    # Set up number of runs and class/plots names
    NumRuns = Params['Splits'][Params['Dataset']]
    plot_name = Params['Dataset'] + ' Test Confusion Matrix'
    avg_plot_name = Params['Dataset'] + ' Test Average Confusion Matrix'
    class_names = Class_names[Params['Dataset']]
    

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
    
    # Initialize arrays for results
    cm_stack = np.zeros((len(class_names), len(class_names)))
    cm_stats = np.zeros((len(class_names), len(class_names), NumRuns))
    FDR_scores = np.zeros((len(class_names), NumRuns))
    log_FDR_scores = np.zeros((len(class_names), NumRuns))
    accuracy = np.zeros(NumRuns)
    MCC = np.zeros(NumRuns)
    
    for split in range(0, numRuns):
        
       
        #Find directory of results
        filename = generate_filename(Params,split)
        if Dataset == 'DeepShip':
            data_dir = process_data(sample_rate=Params['sample_rate'], segment_length=Params['segment_length'])
            data_module = DeepShipDataModule(
                data_dir, Params['batch_size'],
                Params['num_workers'], Params['pin_memory'],
                train_split=0.70, val_split=0.10, test_split=0.20
            )
        else:
            raise ValueError(f'{Dataset} Dataset not found')
            
        print("Preparing data loaders...")
        data_module.prepare_data()
        data_module.setup("fit")
        data_module.setup(stage='test')
        train_loader, val_loader, test_loader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()

        
        # #Load model
        histogram_layer = HistogramLayer(int(num_feature_maps / (feat_map_size * numBins)),
                                         Params['kernel_size'][student_model],
                                         num_bins=numBins, stride=Params['stride'],
                                         normalize_count=Params['normalize_count'],
                                         normalize_bins=Params['normalize_bins'])
    
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
        if mode == 'teacher':
            model_ft = Lightning_Wrapper(
                model.teacher, Params['num_classes'][Dataset], max_iter=len(train_loader),
                label_names=Params['class_names'][Dataset]
            )
        elif mode == 'distillation':
            # pdb.set_trace()
            model.remove_PANN_feature_extractor_teacher()
            sub_dir = generate_filename(Params, split)
            checkpt_path = os.path.join(sub_dir,'tb_logs/model_logs/version_0/checkpoints/best_model_teacher.ckpt')
            
            best_teacher = Lightning_Wrapper.load_from_checkpoint(checkpt_path,
                                                                  hparams_file=os.path.join(sub_dir,'tb_logs/model_logs/version_0/checkpoints/hparams.yaml'),
                                                                  model=nn.Sequential(model.feature_extractor, model.teacher),
                                                                  num_classes = num_classes, max_iter=len(train_loader),
                                            strict=True)
            # pdb.set_trace()
            model.teacher = best_teacher.model[1]

        
            # Remove feature extraction layers from PANN/TIMM
            model.remove_PANN_feature_extractor()
            model.remove_TIMM_feature_extractor()
            
            model_ft = Lightning_Wrapper_KD(model, num_classes=Params['num_classes'][Dataset],  max_iter=len(train_loader),
                                          log_dir = filename, label_names=Params['class_names'][Dataset],
                                          Params=Params,criterion=SSTKAD_Loss(task_num = 4)) 
        elif mode == 'student':
            model_ft = Lightning_Wrapper(
                nn.Sequential(model.feature_extractor, model.student),
                num_classes=num_classes,  
                max_iter=len(train_loader), label_names=Params['class_names'][Dataset],
            )
        else:
            raise RuntimeError(f'{mode} not implemented')
        logger = TensorBoardLogger(filename, version = 'Val_Test', default_hp_metric=False)
        #checkpoint_callback creates dir for saving best model checkpoints
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(filename, 'checkpoints'),
            filename='best_model',
            mode='max',
            monitor='val_accuracy',
        )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=Params['patience'],
            verbose=True,
            mode='min'
        )
        # Initialize the trainer with the custom learning rate finder callback
        trainer = Trainer(callbacks=[checkpoint_callback], 
                          enable_checkpointing = Params['save_results'], 
                          default_root_dir = filename, logger=logger) 

        best_model_path = checkpoint_callback.best_model_path
        pdb.set_trace()


        best_model= Lightning_Wrapper_KD.load_from_checkpoint(
            best_model_path,
            hparams_file=os.path.join(filename, 'tb_logs/model_logs/version_0/checkpoints/hparams.yaml'),
            model= model,
            num_classes=num_classes,max_iter=len(train_loader),
            log_dir = filename,
            strict=True
        )
        print('Testing model...')
        test_results = trainer.test(best_model, dataloaders=test_loader,ckpt_path=best_model_path)

        # print('Testing model...')
        # val_results = trainer.validate(best_model, dataloaders = val_loader)
        # test_results = trainer.test(model_ft, dataloaders=test_loader)
        
        
        
    print('Getting aggregated results...')
    sub_dir = os.path.dirname(sub_dir.rstrip('/'))
    
    aggregation_folder = 'Aggregated_Results/'
    aggregate_and_visualize_confusion_matrices(sub_dir, aggregation_folder, 
                                               dataset=Dataset,label_names=Params['class_names'][Dataset],
                                               figsize=Params['fig_size'], fontsize=Params['font_size'])
    aggregate_tensorboard_logs(sub_dir, aggregation_folder,Dataset)
    print('Aggregated results saved...')

    


def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction, help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/bin16/', help='Location to save models')
    parser.add_argument('--student_model', type=str, default='TDNN', help='Select baseline model architecture')
    parser.add_argument('--teacher_model', type=str, default='MobileNetV1', help='Select baseline model architecture')
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
      