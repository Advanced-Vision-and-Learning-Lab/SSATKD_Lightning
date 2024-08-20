#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:38:19 2024

@author: amir.m
"""
from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import os

import argparse
import torch
import torch.nn as nn

from Demo_Parameters import Parameters

np.float = float  # module 'numpy' has no attribute 'float'
np.int = int  # module 'numpy' has no attribute 'int'
np.object = object  # module 'numpy' has no attribute 'object'
np.bool = bool  # module 'numpy' has no attribute 'bool'
from Utils.Lightning_Wrapper import Lightning_Wrapper, Lightning_Wrapper_KD
from Utils.Save_Results import generate_filename
from Datasets.Get_preprocessed_data import process_data
from DeepShipDataModules import DeepShipDataModule
from Utils.Loss_function import SSTKAD_Loss
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from Utils.RBFHistogramPooling import HistogramLayer
from Utils.Network_functions import initialize_model
def save_tsne_plot(model, test_loader, class_names, device, output_path):
    model.eval()
    model.to(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            features, _ = model(x)
            all_features.append(features.cpu())
            all_labels.append(y.cpu())

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_features)

    # Plot t-SNE
    plt.figure(figsize=(6, 4))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(len(class_names)), colors):
        indices = all_labels == i
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=class_names[i], color=color, alpha=0.7, s=50)  # s=50 for larger points
    plt.xlabel('t-SNE Component 1', fontsize=15)
    plt.ylabel('t-SNE Component 2', fontsize=15)
    plt.title('t-SNE Plot', fontsize=18)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
def save_confusion_matrix(model, test_loader, class_names, device, output_path):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            _, y_pred = model(x)
            all_preds.append(y_pred.cpu().argmax(dim=1))
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=True, annot_kws={"size": 16})

    # Make the annotations more visible
    for t in ax.texts:
        text_value = int(t.get_text())
        t.set_size(16)  # Increase font size
        if text_value > cm.max() / 2:
            t.set_color('white')  # Use white text for dark squares
        else:
            t.set_color('black')  # Use black text for light squares

    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('True', fontsize=15)
    plt.title('Confusion Matrix', fontsize=18)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    
def save_confusion_matrix_with_percentage(model, test_loader, class_names, device, output_path):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            _, y_pred = model(x)
            all_preds.append(y_pred.cpu().argmax(dim=1))
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_percent = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix with percentages
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=True, annot_kws={"size": 16})

    for t in ax.texts:
        text_value = float(t.get_text())
        t.set_size(16)  # Increase font size
        if text_value > cm_percent.max() / 2:
            t.set_color('white')  # Use white text for dark squares
        else:
            t.set_color('black')  # Use black text for light squares

    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('True', fontsize=15)
    plt.title('Confusion Matrix (%)', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
import itertools    



def save_avg_confusion_matrix(model_paths, test_loader, class_names, device, output_path):
    all_cms = []

    for path in model_paths:
        model = LitModel.load_from_checkpoint(checkpoint_path=path)
        model.eval()
        model.to(device)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                _, y_pred = model(x)
                all_preds.append(y_pred.cpu().argmax(dim=1))
                all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Compute confusion matrix for the current run
        cm = confusion_matrix(all_labels, all_preds)
        all_cms.append(cm)

    # Convert to numpy array for easier manipulation
    all_cms = np.array(all_cms)

    # Compute mean and standard deviation of confusion matrices
    mean_cm = np.mean(all_cms, axis=0)
    std_cm = np.std(all_cms, axis=0)
    mean_cm_percent = 100 * mean_cm.astype('float') / mean_cm.sum(axis=1)[:, np.newaxis]
    std_cm_percent = 100 * std_cm.astype('float') / (std_cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    # Plot average confusion matrix with percentages and standard deviations
    plt.figure(figsize=(6, 6))  # Increased figure size
    ax = sns.heatmap(mean_cm_percent, annot=False, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=True, annot_kws={"size": 14}, vmin=0, vmax=100)

    # Increase font size for colorbar values
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    
    for i, j in itertools.product(range(mean_cm.shape[0]), range(mean_cm.shape[1])):
        text_value = f"{int(mean_cm[i, j])} ± {int(std_cm[i, j])}\n({mean_cm_percent[i, j]:.1f} ±\n {std_cm_percent[i, j]:.1f}%)"
        # Dynamically adjust font size if necessary
        font_size = 14 if len(text_value) < 30 else 12  # Adjust size based on content length
        ax.text(j + 0.5, i + 0.5, text_value, ha="center", va="center", fontsize=font_size,
                color="white" if mean_cm_percent[i, j] > 50 else "black")

    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    plt.title('Average Confusion Matrix', fontsize=16)
    plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, fontsize=14)
    plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, fontsize=14)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()

    
    
    
def plot_multiclass_roc(model, test_loader, class_names, device, output_path):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            _, y_pred = model(x)
            all_preds.append(y_pred.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    num_classes = len(class_names)

    # Binarize the output
    all_labels = np.eye(num_classes)[all_labels]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    plt.figure(figsize=(6, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=4,
                 label='{0} (area = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Multi-Class Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right", fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add grid lines
    plt.grid(True)

    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()

    

def extract_scalar_from_events(event_paths, scalar_name):
    scalar_values = []
    for event_path in event_paths:
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()
        if scalar_name in event_acc.Tags()['scalars']:
            scalar_events = event_acc.Scalars(scalar_name)
            values = [event.value for event in scalar_events]
            scalar_values.extend(values)
    return scalar_values


def list_available_scalars(event_paths):
    available_scalars = set()
    for event_path in event_paths:
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()
        available_scalars.update(event_acc.Tags()['scalars'])
    return available_scalars


def save_learning_curves(log_dir, output_path):
    # Get all event files in the log directory
    event_paths = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents.'):
                event_paths.append(os.path.join(root, file))

    # List all available scalars
    available_scalars = list_available_scalars(event_paths)
    print("Available Scalars:", available_scalars)

    # Use correct scalar names after checking available scalars
    train_loss = extract_scalar_from_events(event_paths, 'loss_epoch')
    val_loss = extract_scalar_from_events(event_paths, 'val_loss')

    # Plot learning curves if both scalars are available
    if train_loss and val_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(train_loss, label='Training Loss', color='blue', lw=3)
        plt.plot(val_loss, label='Validation Loss', color='orange', lw=3)
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.title('Learning Curves', fontsize=18)
        plt.legend(loc="best", fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Save the figure
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        print("Required scalars ('loss_epoch', 'val_loss') not found in event files.")


def save_accuracy_curves(log_dir, output_path):
    # Get all event files in the log directory
    event_paths = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents.'):
                event_paths.append(os.path.join(root, file))

    # List all available scalars
    available_scalars = list_available_scalars(event_paths)
    print("Available Scalars:", available_scalars)

    # Use correct scalar names after checking available scalars
    train_acc = extract_scalar_from_events(event_paths, 'train_acc')
    val_acc = extract_scalar_from_events(event_paths, 'val_acc')

    # Plot accuracy curves if both scalars are available
    if train_acc and val_acc:
        plt.figure(figsize=(6, 4))
        plt.plot(train_acc, label='Training Accuracy', color='green', lw=3)
        plt.plot(val_acc, label='Validation Accuracy', color='red', lw=3)
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.title('Accuracy Curves', fontsize=18)
        plt.legend(loc="best", fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Save the figure
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        print("Required scalars ('train_acc', 'val_acc') not found in event files.")



import glob
import pdb
import os


def main(Params):
    Dataset = Params['Dataset']
    student_model = Params['student_model'] 
    teacher_model = Params['teacher_model']
    num_classes = Params['num_classes'][Dataset]
    numRuns = Params['Splits'][Dataset]
    numBins = Params['numBins']
    num_feature_maps = Params['out_channels'][student_model]
    mode = Params['mode']
    feat_map_size = Params['feat_map_size']

    print('\nStarting Experiments...')
    
    numRuns = 3
    s_rate = Params['sample_rate']

    

    
    data_dir = process_data(sample_rate=Params['sample_rate'], segment_length=Params['segment_length'])
    data_module = DeepShipDataModule(
        data_dir, Params['batch_size'],
        Params['num_workers'], Params['pin_memory'],
        train_split=0.70, val_split=0.10, test_split=0.20
    )

    torch.set_float32_matmul_precision('medium')
    # Print the current working directory
    current_directory = os.getcwd()
    filename = os.path.join(current_directory,"Saved_Models/HLTDNN_Fbank_AW/Adagrad/student/Fine_Tuning/DeepShip/TDNN")
    print(f"Current working directory: {current_directory}")
    # tb_logs/model_logs/version_0/checkpoints
    # Collecting model paths for average confusion matrix calculation
    model_paths = []
    # pdb.set_trace()

    for run_number in range(1, numRuns+1):  # Correct range to start from 1
    
    
        histogram_layer = HistogramLayer(
            int(num_feature_maps / (feat_map_size * numBins)),
            Params['kernel_size'][student_model],
            num_bins=numBins, stride=Params['stride'],
            normalize_count=Params['normalize_count'],
            normalize_bins=Params['normalize_bins']
        )


        
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
                model.teacher, Params['num_classes'][Dataset], max_iter=len(train_loader),lr=Params['lr'],
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
                model, num_classes=num_classes,max_iter=len(train_loader),
                label_names=Params['class_names'][Dataset], lr=Params['lr'],
                Params=Params, criterion=SSTKAD_Loss(task_num = 4)
            )
        elif mode == 'student':
            model_ft = Lightning_Wrapper(
                nn.Sequential(model.feature_extractor, model.student),
                num_classes=num_classes,  
                max_iter=len(train_loader), lr=Params['lr'],label_names=Params['class_names'][Dataset],
            )
        else:
            raise RuntimeError(f'{mode} not implemented')
       
        # List the contents of the checkpoints directory
        checkpoints_dir = os.path.join(filename, f"Run_{run_number}/tb_logs/model_logs/version_1/checkpoints")
        best_model_path = os.path.join(checkpoints_dir, "best_model.ckpt")
        if os.path.exists(checkpoints_dir):
            print(f"Contents of {checkpoints_dir}: {os.listdir(checkpoints_dir)}")
        else:
            print(f"Directory does not exist: {checkpoints_dir}")
        
        # Construct the pattern for glob to search for checkpoint files
        pattern = os.path.join(checkpoints_dir, "*.ckpt")
        print(f"Searching for pattern: {pattern}")
        
        # Find files matching the pattern
        matching_files = glob.glob(pattern)
        print(f"Matching files: {matching_files}")  # Print matching files list
        
        # Check if any files were found
        if matching_files:
            best_model_path = matching_files[0]
            model_paths.append(best_model_path)
            print(f"Best model path for run {run_number}: {best_model_path}")
        else:
            print(f"No checkpoint files found for run {run_number}")
    
    print("Model paths found:")
    for path in model_paths:
        print(path)


    # Use the first run's model for individual plots
    best_model_path = model_paths[0]
    run_number = 0

    best_model = Lightning_Wrapper.load_from_checkpoint(
        checkpoint_path=best_model_path,
        Params=Params,
        model=nn.Sequential(model.feature_extractor, model.student),
        num_classes=num_classes,
        Dataset=Dataset,
        pretrained_loaded=True,
        run_number=run_number
    )

    # Get the test dataloader from the data module
    test_loader = data_module.test_dataloader()
    
    # Define class names
    class_names = ["Cargo", "Passengership", "Tanker", "Tug"]
    
    # Create directory if it doesn't exist
    output_dir = f"features/{student_model}_{s_rate}_Run{run_number}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path for the ROC plot
    roc_output_path = os.path.join(output_dir, "roc_curve.png")
    
    # Plot ROC curves and save the figure
    plot_multiclass_roc(best_model, test_loader, class_names, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path=roc_output_path)
        
    # Define output path for the confusion matrix plot with percentages
    #cm_prc_output_path = os.path.join(output_dir, "confusion_matrix_prc.png")
        
    # Save confusion matrix with percentages
    #save_confusion_matrix_with_percentage(best_model, test_loader, class_names, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path=cm_prc_output_path)
    
    
    cm_output_path = os.path.join(output_dir, "confusion_matrix.png")
    save_confusion_matrix(best_model, test_loader, class_names, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path=cm_output_path)
    
    
    # Define output path for the average confusion matrix plot
    avg_cm_output_path = os.path.join(output_dir, "avg_confusion_matrix.png")
    
    # Save average confusion matrix across runs
    save_avg_confusion_matrix(model_paths, test_loader, class_names, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path=avg_cm_output_path)
      
    # Define output path for the t-SNE plot
    #tsne_output_path = os.path.join(output_dir, "tsne_plot.png")
        
    # Save t-SNE plot
    #save_tsne_plot(best_model, test_loader, class_names, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path=tsne_output_path)
     
    
    # Define output path for the learning curves plot
    learning_curves_output_path = os.path.join(output_dir, "learning_curves.png")
    
    # Save learning curves
    log_dir = f"tb_logs/{student_model}_{s_rate}/Run_{run_number}/{student_model}"
    save_learning_curves(log_dir, learning_curves_output_path)
    
    # Define output path for the accuracy curves plot
    accuracy_curves_output_path = os.path.join(output_dir, "accuracy_curves.png")
    
    # Save accuracy curves
    save_accuracy_curves(log_dir, accuracy_curves_output_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction, help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/HLTDNN_Fbank_AW/', help='Location to save models')
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
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256, help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction, help='enables CUDA training')
    parser.add_argument('--audio_feature', type=str, default='STFT', help='Audio feature for extraction')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Select optimizer')
    parser.add_argument('--mixup', type=str, default='True', help='Select data augmenter')
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