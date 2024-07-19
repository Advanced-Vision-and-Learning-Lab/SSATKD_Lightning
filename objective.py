import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import glob
import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from Utils.Lightning_Wrapper import Lightning_Wrapper_KD, Lightning_Wrapper
from DeepShipDataModules import DeepShipDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from Demo_Parameters import Parameters
from Utils.Save_Results import generate_filename_optuna
from Utils.Network_functions import initialize_model
import os
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from Utils.RBFHistogramPooling import HistogramLayer
import pdb
from Datasets.Get_preprocessed_data import process_data
from Utils.Loss_function import SSTKAD_Loss
import csv


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def objective(trial, Params):
    stats_w = trial.suggest_float('stats_w', 0.1, 1)
    struct_w = trial.suggest_float('struct_w', 0.1, 1)
    distill_w = trial.suggest_float('distill_w', 0.1, 1)

    # Ensure the sum of weights equals 1
    weight_sum = stats_w + struct_w + distill_w 
    stats_w /= weight_sum
    struct_w /= weight_sum
    distill_w /= weight_sum

    if Params['HPRC']:
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
    
    for split in range(0, 1):
        set_seeds(split)
        all_val_accs, all_test_accs, all_test_f1s = [], [], []

        histogram_layer = HistogramLayer(
            int(num_feature_maps / (feat_map_size * numBins)),
            Params['kernel_size'][student_model],
            num_bins=numBins, stride=Params['stride'],
            normalize_count=Params['normalize_count'],
            normalize_bins=Params['normalize_bins']
        )

        filename = generate_filename_optuna(Params, split, trial.number)       
        # Create directories if they do not exist
        os.makedirs(filename, exist_ok=True)

        logger = TensorBoardLogger(save_dir=filename, name='tb_logs', version='version_0')
        
        # Log directory
        log_dir = logger.log_dir

        if os.path.exists(log_dir):
            files = glob.glob(f'{log_dir}/events.out.tfevents.*')
            for f in files:
                os.remove(f)
        
        if Dataset == 'DeepShip':
            data_dir = process_data(sample_rate=Params['sample_rate'], segment_length=Params['segment_length'])
            data_module = DeepShipDataModule(
                data_dir,
                Params['batch_size'],
                Params['num_workers'],
                Params['pin_memory'],
                train_split=0.7, val_split=0.1, test_split=0.2
            )
        else:
            raise ValueError('{} Dataset not found'.format(Dataset))
            

        data_module.prepare_data()
        data_module.setup("fit")
        data_module.setup(stage='test')
        train_loader, val_loader, test_loader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()

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
            TDNN_feats=(Params['TDNN_feats'][Dataset]),
            window_length=(Params['window_length'][Dataset]), 
            hop_length=(Params['hop_length'][Dataset]),
            input_feature=Params['feature'],
            sample_rate=Params['sample_rate']
        )

        checkpt_path = '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/CNN/Adagrad/teacher/Pretrained/Fine_Tuning/DeepShip/CNN_14/Run_1/checkpoints/best_model_teacher.ckpt'
        
        best_teacher = Lightning_Wrapper.load_from_checkpoint(
            checkpt_path,
            model=model.teacher,
            num_classes=num_classes, 
            strict=True
        )
        model.teacher = best_teacher.model
        
        # Remove feature extraction layers from PANN/TIMM
        model.remove_PANN_feature_extractor()
        model.remove_TIMM_feature_extractor()
        
        model_ft = Lightning_Wrapper_KD(
            model, num_classes=Params['num_classes'][Dataset], stats_w=0.41,
            struct_w=0.47, distill_w=0.79, max_iter=len(train_loader),
            label_names=Params['class_names'], lr=Params['lr'],
            Params=Params, criterion=SSTKAD_Loss()
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(filename, 'checkpoints_distillation'),
            filename='best_model',
            mode='max',
            monitor='val_accuracy',
            save_top_k=1,
            verbose=True
        )
        
        trainer = Trainer(
            gradient_clip_val=0.5,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=Params['patience']),
                checkpoint_callback,
                TQDMProgressBar(refresh_rate=10)
            ], 
            max_epochs=Params['num_epochs'], 
            enable_checkpointing=Params['save_results'], 
            default_root_dir=filename,
            logger=logger,
        )

        print('Training student model...')
        trainer.fit(model_ft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # val_loss = trainer.callback_metrics['val_loss'].item()
        val_accuracy = trainer.callback_metrics['val_accuracy'].item()

        # Save trial results
        results_file = os.path.join(filename, 'optuna_trial_results.csv')
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([trial.number, val_accuracy, stats_w, struct_w, distill_w])
        print('Testing student model...')
        
        trainer.test(model_ft, dataloaders=test_loader)
        
        return val_accuracy

