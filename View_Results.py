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
    
    for split in range(0, 1):
        
       
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
        # pdb.set_trace()
        train_loader, val_loader, test_loader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()
        # pdb.set_trace()
        
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
            # sub_dir = generate_filename_optuna(Params, split, trial.number)
            checkpt_path = '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/CNN/Adagrad/teacher/Pretrained/Fine_Tuning/DeepShip/CNN_14/Run_1/checkpoints/best_model_teacher.ckpt'
            
            best_teacher = Lightning_Wrapper.load_from_checkpoint(
                checkpt_path,
                model=model.teacher,
                num_classes=num_classes, 
                strict=True
            )
            model.teacher = best_teacher.model
            model.remove_PANN_feature_extractor()
            model.remove_TIMM_feature_extractor()
            
            model_ft = Lightning_Wrapper_KD(
                model, num_classes=num_classes, stats_w=0.41,
                struct_w=0.47, distill_w=0.79, max_iter=len(train_loader),
                label_names=Params['class_names'][Dataset], lr=Params['lr'],
                Params=Params, criterion=SSTKAD_Loss(task_num = 4)
            )
        elif mode == 'student':
            model_ft = Lightning_Wrapper(
                nn.Sequential(model.feature_extractor, model.student),
                num_classes=num_classes,  
                max_iter=len(train_loader), label_names=Params['class_names'][Dataset],
            )
        else:
            raise RuntimeError(f'{mode} not implemented')
        
        #checkpoint_callback creates dir for saving best model checkpoints
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(filename, 'checkpoints'),
            filename='best_model',
            mode='max',
            monitor='val_accuracy',
            save_top_k=1,
            verbose=True,
            save_weights_only=True
        )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=Params['patience'],
            verbose=True,
            mode='min'
        )
        # Initialize the trainer with the custom learning rate finder callback
        trainer = Trainer(
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=Params['patience']),
                checkpoint_callback,
                TQDMProgressBar(refresh_rate=10),
            ], 
            max_epochs=Params['num_epochs'], 
            enable_checkpointing=Params['save_results'], 
            default_root_dir=filename,
        )


        # pdb.set_trace()
        # model.load_state_dict(train_dict['best_model_wts'])
        print('Loading model...')
        best_model_path = os.path.join(filename,"checkpoints/best_model.ckpt")

        print('Testing model...')
        test_results = trainer.test(model_ft, dataloaders=test_loader,ckpt_path=best_model_path)
        
        run_dir = os.path.join(filename,"checkpoints/best_model.ckpt")
        all_metrics = aggregate_metrics(run_dir)
        stats = compute_stats(all_metrics)

    
def extract_metrics(log_dir):
    event_files = [os.path.join(root, file)
                   for root, _, files in os.walk(log_dir)
                   for file in files if 'events.out.tfevents' in file]

    if not event_files:
        print(f"No event files found in {log_dir}.")
        return {}

    metrics = defaultdict(list)
    for event_file in event_files:
        event_acc = EventAccumulator(event_file)
        try:
            event_acc.Reload()
        except Exception as e:
            print(f"Error loading {event_file}: {e}")
            continue

        tags = event_acc.Tags()
        print(f"Tags for {event_file}: {tags}")

        if 'scalars' not in tags:
            continue

        scalar_tags = tags['scalars']
        epochs = {}
        for tag in scalar_tags:
            events = event_acc.Scalars(tag)
            if tag == 'epoch':
                epochs = {e.step: e.value for e in events}
                break

        if not epochs:
            print(f"No epoch tag found in {event_file}.")
            continue

        for tag in scalar_tags:
            if tag != 'epoch':
                events = event_acc.Scalars(tag)
                for e in events:
                    if e.step in epochs:
                        epoch = epochs[e.step]
                        metrics[tag].append((epoch, e.value))

    return metrics

def aggregate_metrics(runs_dirs):
    all_metrics = defaultdict(list)
    for run_dir in runs_dirs:
        print(f"Checking directory: {run_dir}")
        metrics = extract_metrics(run_dir)
        if metrics:
            for key, values in metrics.items():
                all_metrics[key].append(values)
    return all_metrics
def compute_stats(all_metrics):
    stats = {}
    for key, values_list in all_metrics.items():
        all_values = [value for values in values_list for _, value in values]
        if all_values:
            mean = np.mean(all_values)
            std = np.std(all_values)
            stats[key] = {'mean': mean, 'std': std}
    return stats

def plot_train_val_metrics(all_metrics, train_metric_name, val_metric_name, title):
    plt.figure(figsize=(10, 5))
    found_data = False
    for run_idx, (train_values, val_values) in enumerate(zip(all_metrics.get(train_metric_name, []), all_metrics.get(val_metric_name, []))):
        if not train_values or not val_values:
            print(f"No data for {train_metric_name} or {val_metric_name} in run {run_idx + 1}")
            continue
        found_data = True
        train_epochs = [epoch for epoch, _ in train_values]
        train_values = [value for _, value in train_values]
        val_epochs = [epoch for epoch, _ in val_values]
        val_values = [value for _, value in val_values]
        plt.plot(train_epochs, train_values, label=f'Train Run {run_idx + 1}')
        plt.plot(val_epochs, val_values, label=f'Val Run {run_idx + 1}')

    if found_data:
        plt.xlabel('Epochs')
        plt.ylabel(title.replace('_', ' ').title())
        plt.legend()
        plt.title(f'{title.replace("_", " ").title()} Across Runs')
        plt.show()
    else:
        print(f"No data found for {title}.")
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
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256, help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction, help='enables CUDA training')
    parser.add_argument('--audio_feature', type=str, default='Log_Mel_Spectrogram', help='Audio feature for extraction')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Select optimizer')
    parser.add_argument('--mixup', type=str, default='True', help='Select data augmenter')
    parser.add_argument('--patience', type=int, default=25, help='Number of epochs to train each model for (default: 50)')
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
      