# -*- coding: utf-8 -*-
"""
Created on Thursday April 25 22:32:00 2024
Train and evaluate models for experiments on datasets
@author: jpeeples, salimalkharsa
"""
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateFinder
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
from lightning.pytorch.tuner import Tuner
import optuna
import pdb
from Demo_Parameters import Parameters
from Utils.Save_Results import generate_filename
from Utils.Lightning_Wrapper import Lightning_Wrapper, Lightning_Wrapper_KD
from Utils.Network_functions import initialize_model
from DeepShipDataModules import DeepShipDataModule
from Utils.RBFHistogramPooling import HistogramLayer
from Datasets.Get_preprocessed_data import process_data
from Utils.Loss_function import SSTKAD_Loss
from objective import objective
from lightning.pytorch.tuner import Tuner
from thop import profile
from lightning.pytorch.loggers import TensorBoardLogger
from Utils.Save_Results import generate_filename, aggregate_tensorboard_logs, aggregate_and_visualize_confusion_matrices


# from pytorch_lightning.tuner.tuning import LearningRateFinder

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Turn off plotting
plt.ioff()


def set_seeds(seed):
    # pdb.set_trace()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(Params, optimize=False):
    # if Params['HPRC']:
    torch.set_float32_matmul_precision('medium')    
    fig_size = Params['fig_size']
    font_size = Params['font_size']
    plot_name = Params['Dataset'] + ' Test Confusion Matrix'
    avg_plot_name = Params['Dataset'] + ' Test Average Confusion Matrix'

    
    Dataset = Params['Dataset']
    student_model = Params['student_model'] 
    teacher_model = Params['teacher_model']
    num_classes = Params['num_classes'][Dataset]
    numRuns = Params['Splits'][Dataset]
    numBins = Params['numBins']
    num_feature_maps = Params['out_channels'][student_model]
    mode = Params['mode']
    feat_map_size = Params['feat_map_size']


    print('Starting Experiments...')
    best_model_path = ""
    
    for split in range(0, 1):
        set_seeds(split)

        histogram_layer = HistogramLayer(
            int(num_feature_maps / (feat_map_size * numBins)),
            Params['kernel_size'][student_model],
            num_bins=numBins, stride=Params['stride'],
            normalize_count=Params['normalize_count'],
            normalize_bins=Params['normalize_bins']
        )

        filename = generate_filename(Params, split)
        logger = TensorBoardLogger(
            save_dir=os.path.join(filename, "tb_logs"),
            name="model_logs",
        )
        # logger = TensorBoardLogger(filename, default_hp_metric=False, version = 'Training')
        
        #Remove past events to conserve memory allocation
        log_dir = '{}{}/{}'.format(logger.save_dir,logger.name,logger.version)
        # print(f"Model path: {filename}")
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
            
        print("Preparing data loaders...")
        data_module.prepare_data()
        data_module.setup("fit")
        data_module.setup(stage='test')
        # pdb.set_trace()
        train_loader, val_loader, test_loader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()
        # Get class counts for each split
        class_counts = data_module.train_dataset.count_classes_per_split()
        
        print("Train Split Class Counts:", class_counts['train'])
        print("Validation Split Class Counts:", class_counts['val'])
        print("Test Split Class Counts:", class_counts['test'])


        # print(f"Label distribution in test set: {label_counts}")
        print("Dataloaders Initialized.")

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
            sample_rate=Params['sample_rate'],
            level_num = Params['level_num'],
            max_level = Params['max_level'],
        )
        print("Model Initialized.")

        if args.mode == 'teacher':
            sub_dir = generate_filename(Params, split)
            model.remove_PANN_feature_extractor_teacher()
            model_ft = Lightning_Wrapper(
                nn.Sequential(model.feature_extractor, model.teacher), Params['num_classes'][Dataset], max_iter=len(train_loader),lr=Params['lr'],
                label_names=Params['class_names'][Dataset], log_dir =filename,
            )
            
        elif args.mode == 'distillation': 
            sub_dir = generate_filename(Params, split)
            model_ft = Lightning_Wrapper_KD(model, num_classes=Params['num_classes'][Dataset],  max_iter=len(train_loader),lr=Params['lr'],
                                          log_dir = filename, label_names=Params['class_names'][Dataset],
                                          Params=Params,criterion=SSTKAD_Loss(task_num = 4))        
        elif args.mode == 'student':
            sub_dir = generate_filename(Params, split)
            model_ft = Lightning_Wrapper(
                nn.Sequential(model.feature_extractor, model.student),
                num_classes=num_classes,  
                max_iter=len(train_loader), log_dir = filename, lr=Params['lr'],label_names=Params['class_names'][Dataset],
            )
        else:
            raise RuntimeError(f'{mode} not implemented')
            

    print('Getting aggregated results...')
    sub_dir = os.path.dirname(sub_dir.rstrip('/'))
    
    aggregation_folder = 'Aggregated_Results/'
    aggregate_and_visualize_confusion_matrices(sub_dir, aggregation_folder, 
                                               dataset=Dataset,label_names=Params['class_names'][Dataset],
                                               figsize=(30,30), fontsize=42)
    aggregate_tensorboard_logs(sub_dir, aggregation_folder,Dataset)
    print('Aggregated results saved...')

def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction, help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/SSTAKD_paper/', help='Location to save models')
    parser.add_argument('--student_model', type=str, default='TDNN', help='Select baseline model architecture')
    parser.add_argument('--teacher_model', type=str, default='MobileNetV1', help='Select baseline model architecture')
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
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction, help='enables CUDA training')
    parser.add_argument('--audio_feature', type=str, default='Log_Mel_Spectrogram', help='Audio feature for extraction')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Select optimizer')
    parser.add_argument('--ablation', type=str, default='True', help='Select ablation study to be true or false')
    parser.add_argument('--patience', type=int, default=50, help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--level_num', type=int, default=4, help='Number of quantization level for the stat module(default: 8)')
    parser.add_argument('--max_level', type=int, default=3, help='Number of decomposition level for the struct module(default: 3)')
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
