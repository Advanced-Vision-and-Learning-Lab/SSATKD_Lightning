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
# from pytorch_lightning.tuner.tuning import LearningRateFinder

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Turn off plotting
plt.ioff()


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def main(Params, optimize=False):
    # if Params['HPRC']:
    torch.set_float32_matmul_precision('medium')       
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
    
    for split in range(0, numRuns):
        set_seeds(split)
        all_val_accs, all_test_accs, all_test_f1s = [], [], []

        histogram_layer = HistogramLayer(
            int(num_feature_maps / (feat_map_size * numBins)),
            Params['kernel_size'][student_model],
            num_bins=numBins, stride=Params['stride'],
            normalize_count=Params['normalize_count'],
            normalize_bins=Params['normalize_bins']
        )

        filename = generate_filename(Params, split)
        print(f"Model path: {filename}")

        
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
            sample_rate=Params['sample_rate']
        )
        print("Model Initialized.")

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
            
        # model = torch.compile(model_ft, fullgraph=True)
        
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
        
        # Set the logger to save logs in the generated filename directory
        logger = TensorBoardLogger(
            save_dir=os.path.join(filename, "tb_logs"),
            name="model_logs",
        )
        print("Logger set up.")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")

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
            logger=logger,
        )
        # tuner = Tuner(trainer)
        # lr_finder = tuner.lr_find(model_ft,train_dataloaders=train_loader, val_dataloaders=val_loader)
        # # Results can be found in
        # print(lr_finder.results)
        
        # # Plot with
        # fig = lr_finder.plot(suggest=True)
        # fig.savefig('lr_finder_plot.png')  # Save the plot to a file
        # plt.close(fig)  # Close the figure
        
        # # Pick point based on plot, or get suggestion
        # new_lr = lr_finder.suggestion()
        
        # # update hparams of the model
        # model_ft.hparams.lr = new_lr
        
        # print(f"Suggested learning rate: {new_lr}")

        
        print('Training model...')
        trainer.fit(model_ft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        

        # pdb.set_trace()
        # # Load the best model checkpoint
        best_model_path = checkpoint_callback.best_model_path  
    
        # best_model = Lightning_Wrapper_KD.load_from_checkpoint(
        #     checkpoint_path=best_model_path,
        #     Params=Params,
        #     model=model,
        #     num_classes=num_classes,
        #     Dataset=Dataset
        # )
        
        print('Testing model...')
        test_results = trainer.test(model_ft, dataloaders=test_loader,ckpt_path=best_model_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction, help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/demo_stft/', help='Location to save models')
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
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256, help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction, help='enables CUDA training')
    parser.add_argument('--audio_feature', type=str, default='Log_Mel_Spectrogram', help='Audio feature for extraction')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Select optimizer')
    parser.add_argument('--mixup', type=str, default='True', help='Select data augmenter')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to train each model for (default: 50)')
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
